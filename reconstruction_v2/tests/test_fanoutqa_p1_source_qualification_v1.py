from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from assumption_agent.benchmarks import fanoutqa_p1_source_qualification_v1 as q


@pytest.fixture(autouse=True)
def _consumed_analysis_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker_path = tmp_path / "formal-marker.json"
    marker_body = {
        "schema": f"{q.VERSION}_one_shot_marker_v1",
        "status": "started_before_contract_validation_or_dataset_JSON_parse",
        "official_release": q.OFFICIAL_RELEASE,
        "official_tree": q.OFFICIAL_TREE,
        "DEV_git_blob_sha1": q.DEV_GIT_BLOB_SHA1,
        "DEV_size_bytes": q.DEV_SIZE_BYTES,
        "cache_size_bytes": q.CACHE_SIZE_BYTES,
        "test_open_authorized": False,
        "source_item_question_answer_evidence_URL_or_id_output_count": 0,
        "model_action_or_score_count": 0,
        "retry_replay_resample_or_contract_revision": 0,
    }
    q._write_exclusive(
        marker_path,
        {**marker_body, "self_sha256": q._semantic_hash(marker_body)},
    )
    monkeypatch.setattr(q, "MARKER_PATH", marker_path)


def _evidence(pageid: int, revid: int | None = None) -> dict[str, object]:
    return {
        "pageid": pageid,
        "revid": revid if revid is not None else pageid + 1000,
        "title": f"Public title {pageid}",
        "url": f"https://en.wikipedia.org/wiki/Public_title_{pageid}",
    }


def _leaf(node_id: str, pageid: int, depends_on: list[str] | None = None) -> dict[str, object]:
    return {
        "id": node_id,
        "question": f"Public synthetic subquestion {node_id}?",
        "decomposition": [],
        "answer": f"answer-{node_id}",
        "depends_on": list(depends_on or []),
        "evidence": _evidence(pageid),
    }


def _rows() -> list[dict[str, object]]:
    parallel = {
        "id": "public-parallel",
        "question": "Which public synthetic parallel facts are relevant?",
        "decomposition": [
            _leaf("p1", 101),
            _leaf("p2", 102),
            _leaf("p3", 103),
        ],
        "answer": ["a", "b", "c"],
        "categories": ["Geography"],
    }
    dependency = {
        "id": "public-dependency",
        "question": "Which public synthetic dependent facts are relevant?",
        "decomposition": [
            _leaf("d1", 201),
            _leaf("d2", 202, ["d1"]),
            _leaf("d3", 203, ["d2"]),
        ],
        "answer": {"result": "synthetic"},
        "categories": ["History", "Other"],
    }
    hierarchical = {
        "id": "public-hierarchical",
        "question": "Which public synthetic hierarchical facts are relevant?",
        "decomposition": [
            {
                "id": "h0",
                "question": "Resolve the public synthetic hierarchy?",
                "decomposition": [
                    _leaf("h1", 301),
                    _leaf("h2", 302),
                    _leaf("h3", 303),
                ],
                "answer": "synthetic-parent",
                "depends_on": [],
                "evidence": None,
            }
        ],
        "answer": "synthetic-final",
        "categories": ["Technology"],
    }
    return [parallel, dependency, hierarchical]


def _canonical_json(rows: object) -> bytes:
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _make_cache(
    pageids: list[int],
    *,
    unsafe: str | None = None,
    duplicate_directory: bool = False,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        directory = tarfile.TarInfo("wikicache/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        if duplicate_directory:
            archive.addfile(directory)
        for pageid in pageids:
            raw = f"# Public title {pageid}\n\nPublic synthetic article.\n".encode()
            member = tarfile.TarInfo(f"wikicache/{pageid}-dated.md")
            member.size = len(raw)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(raw))
        if unsafe is not None:
            raw = b"unsafe"
            member = tarfile.TarInfo(unsafe)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return output.getvalue()


def _write_fixture(
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
    *,
    cache_pageids: list[int] | None = None,
    unsafe_cache_member: str | None = None,
    duplicate_cache_directory: bool = False,
    required_per_family: int = 1,
) -> tuple[Path, Path, q.QualificationContract]:
    rows = copy.deepcopy(rows if rows is not None else _rows())
    dev_raw = _canonical_json(rows)
    dev = tmp_path / "dev.json"
    dev.write_bytes(dev_raw)
    pageids = cache_pageids or [101, 102, 103, 201, 202, 203, 301, 302, 303]
    cache_raw = _make_cache(
        pageids,
        unsafe=unsafe_cache_member,
        duplicate_directory=duplicate_cache_directory,
    )
    cache = tmp_path / "cache.tar.gz"
    cache.write_bytes(cache_raw)
    contract = q.QualificationContract(
        dev_count=len(rows),
        dev_size_bytes=len(dev_raw),
        dev_git_blob_sha1=q._git_blob_sha1(dev_raw),
        dev_sha256=hashlib.sha256(dev_raw).hexdigest(),
        cache_size_bytes=len(cache_raw),
        required_per_family=required_per_family,
        max_cache_files=100,
        max_cache_uncompressed_bytes=100_000,
    )
    return dev, cache, contract


def test_valid_aggregate_qualification_has_three_structural_families(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["qualified"] is True
    assert result["family_total_counts"] == {
        "HIERARCHICAL": 1,
        "DEPENDENCY_FLAT": 1,
        "PARALLEL_FLAT": 1,
    }
    assert result["page_disjoint_witness_counts"] == {
        "HIERARCHICAL": 1,
        "DEPENDENCY_FLAT": 1,
        "PARALLEL_FLAT": 1,
    }
    assert result["distinct_DEV_evidence_page_count"] == 9
    serialized = json.dumps(result, sort_keys=True)
    assert "Which public" not in serialized
    assert "Public_title" not in serialized


def test_source_sha256_binding_fails_closed(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    bad = q.QualificationContract(
        **{**contract.__dict__, "dev_sha256": "0" * 64}
    )
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="identity"):
        q._analyze_sources(dev, cache, contract=bad)


def test_source_git_blob_binding_fails_closed(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    bad = q.QualificationContract(
        **{**contract.__dict__, "dev_git_blob_sha1": "0" * 40}
    )
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="identity"):
        q._analyze_sources(dev, cache, contract=bad)


def test_exact_top_level_schema_is_required(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["unexpected"] = 1
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="item schema"):
        q._analyze_sources(dev, cache, contract=contract)


def test_unknown_category_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["categories"] = ["Not an official topic"]
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="category"):
        q._analyze_sources(dev, cache, contract=contract)


def test_canonical_wikipedia_query_url_is_accepted(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["decomposition"][0]["evidence"]["url"] = (
        "https://en.wikipedia.org/w/index.php?curid=101"
    )
    dev, cache, contract = _write_fixture(tmp_path, rows)
    assert q._analyze_sources(dev, cache, contract=contract)["qualified"] is True


def test_duplicate_top_level_identity_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["id"] = rows[0]["id"]
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="identity closure"):
        q._analyze_sources(dev, cache, contract=contract)


def test_duplicate_normalized_question_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["question"] = str(rows[0]["question"]).upper()
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="identity closure"):
        q._analyze_sources(dev, cache, contract=contract)


def test_missing_dependency_reference_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["decomposition"][1]["depends_on"] = ["absent"]
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="reference closure"):
        q._analyze_sources(dev, cache, contract=contract)


def test_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["decomposition"][0]["depends_on"] = ["d2"]
    rows[1]["decomposition"][1]["depends_on"] = ["d1"]
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="cyclic"):
        q._analyze_sources(dev, cache, contract=contract)


def test_internal_node_requires_children_and_no_evidence(tmp_path: Path) -> None:
    rows = _rows()
    rows[2]["decomposition"][0]["evidence"] = _evidence(399)
    dev, cache, contract = _write_fixture(tmp_path, rows)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="leaf evidence"):
        q._analyze_sources(dev, cache, contract=contract)


def test_conflicting_revision_is_aggregate_ineligible_not_source_invalid(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["decomposition"][2]["evidence"] = _evidence(101, 999999)
    dev, cache, contract = _write_fixture(tmp_path, rows, required_per_family=1)
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["eligible_family_counts"]["PARALLEL_FLAT"] == 0
    assert result["qualified"] is False


def test_cross_item_revision_conflict_is_global_ineligibility(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["decomposition"][0]["evidence"] = _evidence(101, 777777)
    dev, cache, contract = _write_fixture(tmp_path, rows)
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["eligible_family_counts"]["PARALLEL_FLAT"] == 0
    assert result["eligible_family_counts"]["DEPENDENCY_FLAT"] == 0
    assert result["qualified"] is False


def test_cache_missing_page_makes_item_ineligible(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(
        tmp_path,
        cache_pageids=[101, 102, 201, 202, 203, 301, 302, 303],
    )
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["eligible_family_counts"]["PARALLEL_FLAT"] == 0
    assert result["qualified"] is False


@pytest.mark.parametrize("member", ["../escape", "/absolute", "other/1-dated.md"])
def test_unsafe_or_noncanonical_cache_member_is_rejected(
    tmp_path: Path, member: str
) -> None:
    dev, cache, contract = _write_fixture(tmp_path, unsafe_cache_member=member)
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="archive"):
        q._analyze_sources(dev, cache, contract=contract)


def test_cache_size_binding_is_required(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    bad = q.QualificationContract(
        **{**contract.__dict__, "cache_size_bytes": contract.cache_size_bytes + 1}
    )
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="bound regular"):
        q._analyze_sources(dev, cache, contract=bad)


def test_duplicate_cache_directory_name_is_rejected(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(
        tmp_path, duplicate_cache_directory=True
    )
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="duplicated"):
        q._analyze_sources(dev, cache, contract=contract)


@pytest.mark.parametrize("mutation", ["crc", "truncated_footer", "trailing_payload"])
def test_complete_gzip_envelope_is_required(tmp_path: Path, mutation: str) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    raw = bytearray(cache.read_bytes())
    if mutation == "crc":
        raw[-8] ^= 0x01
    elif mutation == "truncated_footer":
        del raw[-8:]
    else:
        raw.extend(b"not-a-gzip-member")
    cache.write_bytes(raw)
    mutated = q.QualificationContract(
        **{**contract.__dict__, "cache_size_bytes": len(raw)}
    )
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="archive"):
        q._analyze_sources(dev, cache, contract=mutated)


def test_evidence_count_outside_frozen_range_is_ineligible(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["decomposition"] = rows[0]["decomposition"][:2]
    dev, cache, contract = _write_fixture(tmp_path, rows)
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["eligible_family_counts"]["PARALLEL_FLAT"] == 0
    assert result["qualified"] is False


def test_paper_example_hash_is_excluded_without_membership_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    digest = q._normalized_question_sha256(_rows()[0]["question"])
    monkeypatch.setattr(q, "EXAMPLE_QUESTION_DENY_SHA256", frozenset({digest}))
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["eligible_family_counts"]["PARALLEL_FLAT"] == 0
    serialized = json.dumps(result, sort_keys=True)
    assert digest not in serialized
    assert "deny" not in serialized.casefold()
    assert result["qualified"] is False


def test_global_page_overlap_can_fail_disjoint_capacity(tmp_path: Path) -> None:
    rows = _rows()
    rows[1]["decomposition"][0]["evidence"] = _evidence(101)
    dev, cache, contract = _write_fixture(tmp_path, rows)
    result = q._analyze_sources(dev, cache, contract=contract)
    assert result["qualified"] is False
    assert sum(result["page_disjoint_witness_counts"].values()) == 2


def test_witness_search_repairs_simple_greedy_false_negative() -> None:
    candidates: list[q.Candidate] = []
    for family_index, family in enumerate(q.FAMILIES):
        offset = family_index * 10
        candidates.extend(
            [
                q.Candidate(
                    hashlib.sha256(f"{family}-wide".encode()).hexdigest(),
                    family,
                    frozenset({offset + 1, offset + 2}),
                ),
                q.Candidate(
                    hashlib.sha256(f"{family}-left".encode()).hexdigest(),
                    family,
                    frozenset({offset + 1}),
                ),
                q.Candidate(
                    hashlib.sha256(f"{family}-right".encode()).hexdigest(),
                    family,
                    frozenset({offset + 2}),
                ),
            ]
        )
    assert q._page_disjoint_witness(candidates, 2) == {
        "HIERARCHICAL": 2,
        "DEPENDENCY_FLAT": 2,
        "PARALLEL_FLAT": 2,
    }


def test_analysis_requires_the_consumed_fixed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    monkeypatch.setattr(q, "MARKER_PATH", tmp_path / "missing-marker.json")
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="marker"):
        q._analyze_sources(dev, cache, contract=contract)


def test_aggregate_output_matches_custody_whitelist(tmp_path: Path) -> None:
    dev, cache, contract = _write_fixture(tmp_path)
    result = q._analyze_sources(dev, cache, contract=contract)
    assert set(result) == {
        "allowed_categories",
        "cache_aggregate",
        "cache_covered_distinct_DEV_evidence_page_count",
        "category_counts",
        "DEV_git_blob_sha1",
        "DEV_row_count",
        "DEV_sha256",
        "DEV_size_bytes",
        "distinct_DEV_evidence_page_count",
        "eligible_family_counts",
        "family_total_counts",
        "page_disjoint_witness_counts",
        "qualified",
        "required_page_disjoint_witness_per_family",
        "schema_contract",
    }


def test_canonical_receipt_is_exclusive_and_self_hashable(tmp_path: Path) -> None:
    body = {"schema": "public_test", "value": 1}
    payload = {**body, "self_sha256": q._semantic_hash(body)}
    path = tmp_path / "receipt.json"
    digest = q._write_exclusive(path, payload)
    assert digest == hashlib.sha256(q._canonical_bytes(payload)).hexdigest()
    loaded = q._load_canonical_self_hashed(path, "test receipt")
    assert loaded == payload
    with pytest.raises(q.FanOutQaP1SourceQualificationError, match="consumed"):
        q._write_exclusive(path, payload)
