from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import semantic_assignment_oa_feasibility_v1 as oa
from assumption_agent.models import stable_hash


def _stratum() -> dict[str, object]:
    return {
        "stratum": "DNA",
        "expected_destination": "DNA",
        "search": "DNA genomics",
        "from_publication_date": "2020-01-01",
        "to_publication_date": "2025-12-31",
        "required_text_all_regex": [r"\bDNA\b"],
        "required_text_any_regex": [],
        "required_topic_any_regex": [r"genom|genetic|molecular biology"],
        "excluded_text_any_regex": [],
    }


def _work(identifier: str = "W123") -> dict[str, object]:
    return {
        "id": f"https://openalex.org/{identifier}",
        "type": "article",
        "is_retracted": False,
        "publication_date": "2024-04-03",
        "title": "Reliable DNA sequence analysis",
        "abstract_inverted_index": {
            "genomic": [0],
            "methods": [1],
            "for": [2],
            "DNA": [3],
        },
        "primary_topic": {"display_name": "Genomics and genetic variation"},
        "topics": [{"display_name": "Molecular biology"}],
        "concepts": [],
        "doi": "https://doi.org/10.1/example",
        "ids": {"openalex": f"https://openalex.org/{identifier}"},
        "locations": [],
        "best_oa_location": {
            "is_oa": True,
            "pdf_url": f"https://publisher.example/{identifier}.pdf",
            "license": "cc-by",
        },
    }


def test_metadata_predicate_is_label_first_and_excludes_train_arxiv() -> None:
    candidate = oa.metadata_candidate(
        _work(), stratum=_stratum(), known_train_arxiv_ids=set()
    )
    assert candidate is not None
    assert candidate["expected_destination"] == "DNA"
    assert candidate["openalex_id"] == "W123"

    contaminated = _work()
    contaminated["locations"] = [
        {"landing_page_url": "https://arxiv.org/abs/1909.05563v1"}
    ]
    assert (
        oa.metadata_candidate(
            contaminated,
            stratum=_stratum(),
            known_train_arxiv_ids={"1909.05563"},
        )
        is None
    )

    arxiv_pdf = _work()
    arxiv_pdf["best_oa_location"]["pdf_url"] = (
        "https://arxiv.org/pdf/2601.00001"
    )
    assert (
        oa.metadata_candidate(
            arxiv_pdf, stratum=_stratum(), known_train_arxiv_ids=set()
        )
        is None
    )


def test_sampling_rank_is_hash_deterministic_and_order_independent() -> None:
    works = [_work("W3"), _work("W1"), _work("W2")]
    first = oa.rank_metadata_candidates(
        works,
        stratum=_stratum(),
        seed="fixed-seed",
        known_train_arxiv_ids=set(),
    )
    second = oa.rank_metadata_candidates(
        reversed(works),
        stratum=_stratum(),
        seed="fixed-seed",
        known_train_arxiv_ids=set(),
    )
    assert [row["openalex_id"] for row in first] == [
        row["openalex_id"] for row in second
    ]
    assert [row["sampling_rank"] for row in first] == [0, 1, 2]
    assert all("title" not in row and "abstract" not in row for row in first)


def test_doi_and_download_task_identities_are_canonical() -> None:
    assert oa._canonical_doi(" https://doi.org/10.1000/ABC ") == "10.1000/abc"
    assert oa._canonical_doi("doi:10.1000/abc") == "10.1000/abc"
    assert oa._canonical_doi("not-a-doi") is None
    common = {
        "openalex_id": "W123",
        "sampling_hash": "a" * 64,
    }
    first = oa._download_task_identity({**common, "stratum": "DNA"})
    second = oa._download_task_identity(
        {**common, "stratum": "unrelated_public_default"}
    )
    assert first != second


def test_malformed_or_nonpublic_pdf_urls_fail_closed() -> None:
    for value in (
        "https://[broken",
        "https://example.org:99999/file.pdf",
        "https://example.org/bad path.pdf",
        "http://example.org/file.pdf",
        "https://127.0.0.1/file.pdf",
        "https://arxiv.org/pdf/1234.56789",
    ):
        with pytest.raises(oa.OaFeasibilityError):
            oa._safe_public_https_url(value)


def test_locked_pack_reopens_hashes_without_persisting_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    expected = {
        "LLM": "LLM",
        "trapped_ion_and_qc": "trapped_ion_and_qc",
        "black_hole": "black_hole",
        "DNA": "DNA",
        "music_history": "music_history",
        "unrelated_public_default": "music_history",
    }
    records = []
    texts_by_name = {}
    content_hashes = []
    for stratum_index, stratum in enumerate(oa.EXPECTED_STRATA):
        for rank in range(10):
            raw = f"%PDF-1.4\npublic test {stratum_index} {rank}".encode()
            digest = hashlib.sha256(raw).hexdigest()
            name = f"{digest}.pdf"
            (pdf_root / name).write_bytes(raw)
            text = f"bounded evidence {stratum_index} {rank}"
            texts_by_name[name] = text
            content_hashes.append(digest)
            records.append(
                {
                    "record_id": hashlib.sha256(
                        f"record-{stratum_index}-{rank}".encode()
                    ).hexdigest(),
                    "openalex_id_hash": hashlib.sha256(
                        f"work-{stratum_index}-{rank}".encode()
                    ).hexdigest(),
                    "doi_hash": hashlib.sha256(
                        f"doi-{stratum_index}-{rank}".encode()
                    ).hexdigest(),
                    "license_hash": "3" * 64,
                    "source_url_hash": hashlib.sha256(
                        f"url-{stratum_index}-{rank}".encode()
                    ).hexdigest(),
                    "stratum": stratum,
                    "expected_destination": expected[stratum],
                    "sampling_rank": rank,
                    "pdf_relative_path": f"pdfs/{name}",
                    "pdf_sha256": digest,
                    "size_bytes": len(raw),
                    "evidence_text_sha256": hashlib.sha256(
                        text.encode()
                    ).hexdigest(),
                }
            )
    pack: dict[str, object] = {
        "pack_version": oa.OA_PACK_VERSION,
        "preregistration_manifest_hash": "5" * 64,
        "candidate_id": "6" * 64,
        "operator_asset_manifest_hash": "7" * 64,
        "seed": "seed",
        "record_count": 60,
        "records_per_stratum": 10,
        "strata": list(oa.EXPECTED_STRATA),
        "records": records,
        "records_hash": stable_hash(records),
        "selection_hash": stable_hash(records),
        "pdf_content_set_hash": stable_hash(sorted(content_hashes)),
        "metadata_query_hashes": {
            stratum: hashlib.sha256(f"query-{stratum}".encode()).hexdigest()
            for stratum in oa.EXPECTED_STRATA
        },
        "metadata_result_set_hashes": {
            stratum: hashlib.sha256(f"metadata-{stratum}".encode()).hexdigest()
            for stratum in oa.EXPECTED_STRATA
        },
        "qualified_candidate_set_hashes": {
            stratum: hashlib.sha256(f"qualified-{stratum}".encode()).hexdigest()
            for stratum in oa.EXPECTED_STRATA
        },
        "download_attempt_count": 0,
        "download_attempt_ledger": [],
        "download_attempt_ledger_hash": stable_hash([]),
        "selection_ledger": [
            {
                "stratum": row["stratum"],
                "sampling_hash": hashlib.sha256(
                    f"sample-{row['stratum']}-{row['sampling_rank']}".encode()
                ).hexdigest(),
                "sampling_rank": row["sampling_rank"],
                "openalex_id_hash": row["openalex_id_hash"],
                "doi_hash": row["doi_hash"],
                "disposition": "selected",
            }
            for row in records
        ],
        "transport_failure_counts": {},
        "prediction_started": False,
        "semantic_outcome_used_for_selection": False,
        "operator_created_extracted_text_artifact": False,
        "raw_title_abstract_or_text_persisted": False,
        "acquisition_online_calls_only": True,
    }
    pack["selection_ledger_hash"] = stable_hash(pack["selection_ledger"])
    pack["selection_disposition_counts"] = {"selected": 60}
    pack["download_attempt_ledger"] = [
        {
            "stratum": row["stratum"],
            "sampling_hash": row["sampling_hash"],
            "sampling_rank": row["sampling_rank"],
            "openalex_id_hash": row["openalex_id_hash"],
            "doi_hash": row["doi_hash"],
            "transport_valid": True,
            "error_type": None,
            "pdf_sha256": records[index]["pdf_sha256"],
            "final_url_hash": records[index]["source_url_hash"],
        }
        for index, row in enumerate(pack["selection_ledger"])
    ]
    pack["download_attempt_count"] = 60
    pack["download_attempt_ledger_hash"] = stable_hash(
        pack["download_attempt_ledger"]
    )
    pack["pack_hash"] = stable_hash(pack)
    monkeypatch.setattr(oa, "_extract_pdf", lambda path: texts_by_name[path.name])

    serialized_pack = json.loads(json.dumps(pack, sort_keys=True))
    reopened, texts = oa.verify_locked_pack(
        serialized_pack,
        pack_root=tmp_path,
        forbidden_train_pdf_hashes=set(),
    )

    assert reopened == records
    assert len(texts) == 60
    assert all(text not in json.dumps(pack, sort_keys=True) for text in texts)

    tampered = copy.deepcopy(serialized_pack)
    tampered["selection_ledger"][0]["disposition"] = "dedupe_doi"
    tampered["selection_ledger_hash"] = stable_hash(tampered["selection_ledger"])
    tampered["selection_disposition_counts"] = {
        "dedupe_doi": 1,
        "selected": 59,
    }
    tampered_body = dict(tampered)
    tampered_body.pop("pack_hash")
    tampered["pack_hash"] = stable_hash(tampered_body)
    with pytest.raises(oa.OaFeasibilityError, match="cannot be replayed"):
        oa.verify_locked_pack(
            tampered,
            pack_root=tmp_path,
            forbidden_train_pdf_hashes=set(),
        )


def test_decision_lock_is_exclusive_and_reserved_failure_is_consumed(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "decision.lock.json"
    oa._reserve_decision_lock(
        lock,
        preregistration_hash="a" * 64,
        pack_hash="b" * 64,
        candidate_id="c" * 64,
    )
    payload = json.loads(lock.read_text())
    assert payload["state"] == "reserved"
    with pytest.raises(FileExistsError, match="already consumed"):
        oa._reserve_decision_lock(
            lock,
            preregistration_hash="a" * 64,
            pack_hash="b" * 64,
            candidate_id="c" * 64,
        )


def test_metadata_semantic_change_is_not_a_transport_replacement() -> None:
    wrong_topic = _work()
    wrong_topic["primary_topic"] = {"display_name": "Civil engineering"}
    wrong_topic["topics"] = []
    assert (
        oa.metadata_candidate(
            wrong_topic,
            stratum=_stratum(),
            known_train_arxiv_ids=set(),
        )
        is None
    )

    missing_pdf = _work()
    missing_pdf["best_oa_location"]["pdf_url"] = None
    assert (
        oa.metadata_candidate(
            missing_pdf,
            stratum=_stratum(),
            known_train_arxiv_ids=set(),
        )
        is None
    )


def test_repository_preregistration_binds_one_exact_offline_decision() -> None:
    root = Path(__file__).resolve().parents[1]
    preregistration = oa.load_preregistration(
        root / "manifests/semantic_assignment_public_oa_feasibility_v1.json"
    )
    assert preregistration["decision_budget"] == 1
    assert preregistration["evaluation"]["required_evidence_valid"] == 60
    assert preregistration["evaluation"]["required_correct"] == 60
    assert preregistration["evaluation"]["ruoli_calls"] == 0
    assert preregistration["operator_freeze"][
        "target_stratum_metadata_or_outcomes_observed_before_freeze"
    ] is False
