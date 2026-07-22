from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import birco_p1_source_qualification_v1 as q


FAMILIES = ("doris-mae", "clinical-trial", "wtb")


def _family(name: str) -> dict[str, object]:
    queries = {
        f"{name}-q0": f"Private synthetic query zero for {name}",
        f"{name}-q1": f"Private synthetic query one for {name}",
    }
    corpus = {
        f"{name}-d0": f"Private synthetic document zero for {name}",
        f"{name}-d1": f"Private synthetic document one for {name}",
        f"{name}-d2": f"Private synthetic document two for {name}",
        f"{name}-d3": f"Private synthetic document three for {name}",
    }
    if name == "doris-mae":
        scores0 = [2.0, 0.5]
        scores1 = [1.5, 0.0]
    else:
        scores0 = [2.0, 0.0] if name == "clinical-trial" else [1.0, 0.0]
        scores1 = [1.0, 0.0]
    return {
        "query": queries,
        "corpus": corpus,
        "qrel": {
            f"{name}-q0": {
                f"{name}-d0": scores0[0],
                f"{name}-d1": scores0[1],
            },
            f"{name}-q1": {
                f"{name}-d1": scores1[0],
                f"{name}-d2": scores1[1],
            },
        },
    }


def _source() -> dict[str, object]:
    return {name: _family(name) for name in FAMILIES}


def _write_source(tmp_path: Path, value: object) -> tuple[Path, q.QualificationContract]:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "BIRCO_dataset.json"
    path.write_bytes(raw)
    contract = q.QualificationContract(
        source_size_bytes=len(raw),
        source_md5=hashlib.md5(raw).hexdigest(),  # nosec B303: fixture identity
        source_sha256=hashlib.sha256(raw).hexdigest(),
        families={
            "doris-mae": q.FamilyContract(2, 4, None),
            "clinical-trial": q.FamilyContract(2, 4, (0.0, 1.0, 2.0)),
            "wtb": q.FamilyContract(2, 4, (0.0, 1.0)),
        },
        minimum_pool_size=2,
        selected_query_capacity=2,
    )
    return path, contract


def test_valid_source_emits_aggregates_without_row_leakage(tmp_path: Path) -> None:
    path, contract = _write_source(tmp_path, _source())
    result = q._analyze_source(path, contract)
    assert result["qualified"] is True
    assert result["family_aggregates"]["wtb"]["query_count"] == 2
    assert result["family_aggregates"]["wtb"]["candidate_membership"] == {
        "distinct_candidate_count": 3,
        "maximum_pool_size": 2,
        "minimum_pool_size": 2,
        "pool_entry_count": 4,
    }
    serialized = json.dumps(result, sort_keys=True)
    assert "Private synthetic" not in serialized
    assert "wtb-q0" not in serialized
    assert "wtb-d0" not in serialized


def test_overlap_components_are_aggregate_only(tmp_path: Path) -> None:
    value = _source()
    path, contract = _write_source(tmp_path, value)
    components = q._analyze_source(path, contract)["family_aggregates"]["wtb"][
        "candidate_overlap_components"
    ]
    assert components == {
        "component_count": 1,
        "largest_component_query_count": 2,
        "singleton_component_count": 0,
    }


def test_pinned_schema_tolerates_unconsumed_extra_fields(tmp_path: Path) -> None:
    value = _source()
    value["unconsumed-metadata"] = {"version": 1}
    value["wtb"]["unconsumed"] = [1, 2, 3]
    path, contract = _write_source(tmp_path, value)
    assert q._analyze_source(path, contract)["qualified"] is True


def test_unreferenced_corpus_text_does_not_create_a_false_terminal(tmp_path: Path) -> None:
    value = _source()
    value["wtb"]["corpus"]["wtb-d3"] = ""
    path, contract = _write_source(tmp_path, value)
    assert q._analyze_source(path, contract)["qualified"] is True


def test_duplicate_json_object_key_is_rejected() -> None:
    with pytest.raises(q.BircoP1SourceQualificationError, match="duplicate"):
        q._parse_json(b'{"x":1,"x":2}')


def test_nonfinite_json_number_is_rejected() -> None:
    with pytest.raises(q.BircoP1SourceQualificationError, match="non-finite"):
        q._parse_json(b'{"x":NaN}')


def test_source_sha256_identity_is_required(tmp_path: Path) -> None:
    path, contract = _write_source(tmp_path, _source())
    bad = q.QualificationContract(
        **{**contract.__dict__, "source_sha256": "0" * 64}
    )
    with pytest.raises(q.BircoP1SourceQualificationError, match="SHA256"):
        q._analyze_source(path, bad)


def test_source_md5_identity_is_required(tmp_path: Path) -> None:
    path, contract = _write_source(tmp_path, _source())
    bad = q.QualificationContract(**{**contract.__dict__, "source_md5": "0" * 32})
    with pytest.raises(q.BircoP1SourceQualificationError, match="MD5"):
        q._analyze_source(path, bad)


def test_required_family_field_is_required(tmp_path: Path) -> None:
    value = _source()
    del value["wtb"]["qrel"]
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="required"):
        q._analyze_source(path, contract)


def test_qrel_query_identity_set_must_match(tmp_path: Path) -> None:
    value = _source()
    del value["wtb"]["qrel"]["wtb-q1"]
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="identity sets"):
        q._analyze_source(path, contract)


def test_candidate_must_exist_in_family_corpus(tmp_path: Path) -> None:
    value = _source()
    value["wtb"]["qrel"]["wtb-q0"]["absent-document"] = 0
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="absent"):
        q._analyze_source(path, contract)


@pytest.mark.parametrize("score", [True, "1", -1, 3, float("inf")])
def test_invalid_qrel_score_is_rejected(tmp_path: Path, score: object) -> None:
    value = _source()
    value["wtb"]["qrel"]["wtb-q0"]["wtb-d0"] = score
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="score|non-finite"):
        q._analyze_source(path, contract)


def test_wtb_discrete_score_domain_is_enforced(tmp_path: Path) -> None:
    value = _source()
    value["wtb"]["qrel"]["wtb-q0"]["wtb-d0"] = 0.5
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="discrete"):
        q._analyze_source(path, contract)


def test_minimum_candidate_pool_is_enforced(tmp_path: Path) -> None:
    value = _source()
    del value["wtb"]["qrel"]["wtb-q0"]["wtb-d1"]
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="minimum"):
        q._analyze_source(path, contract)


def test_each_candidate_pool_requires_positive_gain(tmp_path: Path) -> None:
    value = _source()
    value["wtb"]["qrel"]["wtb-q0"] = {"wtb-d0": 0, "wtb-d1": 0}
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="positive"):
        q._analyze_source(path, contract)


def test_query_capacity_is_enforced(tmp_path: Path) -> None:
    path, contract = _write_source(tmp_path, _source())
    bad = q.QualificationContract(
        **{**contract.__dict__, "selected_query_capacity": 3}
    )
    with pytest.raises(q.BircoP1SourceQualificationError, match="capacity"):
        q._analyze_source(path, bad)


def test_one_shot_marker_cannot_be_consumed_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(q, "MARKER_PATH", tmp_path / "marker.json")
    first = q._consume_marker()
    assert len(first) == 64
    with pytest.raises(FileExistsError):
        q._consume_marker()


def test_exclusive_result_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    q._write_exclusive(path, {"schema": "synthetic"})
    with pytest.raises(FileExistsError):
        q._write_exclusive(path, {"schema": "replacement"})
    assert json.loads(path.read_text()) == {"schema": "synthetic"}


def test_fixture_mutation_does_not_hide_schema_errors(tmp_path: Path) -> None:
    value = copy.deepcopy(_source())
    value["wtb"]["query"]["wtb-q0"] = ""
    path, contract = _write_source(tmp_path, value)
    with pytest.raises(q.BircoP1SourceQualificationError, match="text"):
        q._analyze_source(path, contract)
