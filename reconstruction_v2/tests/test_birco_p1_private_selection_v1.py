from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks import birco_p1_private_selection_v1 as selection
from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic_contract


OBJECTIVES = {
    family: f"PRIVATE_OBJECTIVE_SENTINEL::{family}"
    for family in selection.FAMILIES
}
SECRET = bytes(range(32))


@dataclass(frozen=True)
class SyntheticSource:
    value: dict[str, Any]
    raw: bytes
    contract: selection.SourceContract


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _synthetic_source(*, query_count: int = 40, corpus_count: int = 10) -> SyntheticSource:
    value: dict[str, Any] = {}
    family_contracts: list[selection.FamilyContract] = []
    for family in selection.FAMILIES:
        corpus = {
            f"PRIVATE_CID::{family}::{candidate_i:02d}": (
                f"PRIVATE_DOCUMENT_TEXT::{family}::{candidate_i:02d}. "
                + " ".join(
                    f"Evidence clause {clause_i:02d} for {family} candidate "
                    f"{candidate_i:02d}."
                    for clause_i in range(12)
                )
            )
            for candidate_i in range(corpus_count)
        }
        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, float]] = {}
        candidate_ids = tuple(corpus)
        for query_i in range(query_count):
            qid = f"PRIVATE_QID::{family}::{query_i:03d}"
            queries[qid] = f"PRIVATE_QUERY_TEXT::{family}::{query_i:03d}"
            if family == "doris-mae":
                positive = 1.5
            elif family == "clinical-trial":
                positive = 2.0
            else:
                positive = 1.0
            qrels[qid] = {
                cid: positive if candidate_i == query_i % corpus_count else 0.0
                for candidate_i, cid in enumerate(candidate_ids)
            }
        value[family] = {
            "query": queries,
            "corpus": corpus,
            "qrel": qrels,
            "unconsumed_synthetic_metadata": {"allowed": True},
        }
        allowed = (
            None
            if family == "doris-mae"
            else (0.0, 1.0, 2.0)
            if family == "clinical-trial"
            else (0.0, 1.0)
        )
        family_contracts.append(
            selection.FamilyContract(
                family,
                query_count,
                corpus_count,
                allowed,
            )
        )
    raw = _canonical(value)
    contract = selection.SourceContract(
        source_size_bytes=len(raw),
        source_md5=hashlib.md5(raw).hexdigest(),  # nosec: synthetic byte identity
        source_sha256=hashlib.sha256(raw).hexdigest(),
        families=tuple(family_contracts),
        minimum_pool_size=corpus_count,
    )
    return SyntheticSource(value=value, raw=raw, contract=contract)


def _qualification(
    contract: selection.SourceContract,
) -> dict[str, Any]:
    body = {
        "schema": "birco_p1_source_qualification_v1_result_v1",
        "status": "qualified_aggregate_only",
        "qualified": True,
        "model_action_or_score_count": 0,
        "online_evaluator_call_count": 0,
        "qrel_value_output_count": 0,
        "source_identity": {
            "md5": contract.source_md5,
            "sha256": contract.source_sha256,
            "size_bytes": contract.source_size_bytes,
        },
        "family_aggregates": {
            family.name: {
                "query_count": family.query_count,
                "corpus_count": family.corpus_count,
                "query_disjoint_selected_capacity": selection.SELECTED_PER_FAMILY,
                "candidate_membership": {
                    "distinct_candidate_count": family.corpus_count,
                    "minimum_pool_size": contract.minimum_pool_size,
                },
            }
            for family in contract.families
        },
    }
    return selection.self_hashed(body, "self_sha256")


def _write_fixture(tmp_path: Path, source: SyntheticSource) -> tuple[Path, Path, str]:
    source_path = tmp_path / "BIRCO.synthetic.json"
    source_path.write_bytes(source.raw)
    source_path.chmod(0o600)
    qualification = _qualification(source.contract)
    qualification_path = tmp_path / "qualification.synthetic.json"
    qualification_path.write_bytes(_canonical(qualification))
    return qualification_path, source_path, str(qualification["self_sha256"])


def _acquire(
    tmp_path: Path,
    *,
    root_name: str = "selection",
    random_bytes: Any = None,
) -> tuple[dict[str, Any], Path, SyntheticSource, Path, Path, str]:
    source = _synthetic_source()
    qualification_path, source_path, qualification_sha = _write_fixture(
        tmp_path, source
    )
    output_root = tmp_path / root_name
    receipt = selection.acquire_once(
        source_path=source_path,
        qualification_path=qualification_path,
        output_root=output_root,
        contract=source.contract,
        expected_qualification_self_sha256=qualification_sha,
        task_objectives=OBJECTIVES,
        random_bytes=(lambda size: SECRET) if random_bytes is None else random_bytes,
    )
    return (
        receipt,
        output_root,
        source,
        qualification_path,
        source_path,
        qualification_sha,
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _manual_frame(name: bytes, value: str) -> bytes:
    raw = value.encode("utf-8")
    return name + b"\0" + len(raw).to_bytes(8, "big") + raw


def test_exact_hmac_order_is_deterministic_and_uses_fixed_windows() -> None:
    source = _synthetic_source(query_count=45)
    records = selection.validate_source_payload(source.value, contract=source.contract)
    selected = selection.select_private_blocks(records, secret=SECRET)
    selected_again = selection.select_private_blocks(records, secret=SECRET)
    assert selected == selected_again

    for family in selection.FAMILIES:
        qids = [row.qid for row in records[family]]
        expected = sorted(
            qids,
            key=lambda qid: (
                hmac.new(
                    SECRET,
                    selection.ORDER_HMAC_DOMAIN
                    + _manual_frame(b"study", selection.STUDY_ID)
                    + _manual_frame(b"family", family)
                    + _manual_frame(b"qid", qid),
                    hashlib.sha256,
                ).digest(),
                qid.encode("utf-8"),
            ),
        )[: selection.SELECTED_PER_FAMILY]
        observed = [
            row.qid
            for block in selection.BLOCK_ORDER
            for row in selected[block]
            if row.family == family
        ]
        assert observed == expected
        for block in selection.BLOCK_ORDER:
            assert sum(row.family == family for row in selected[block]) == 10
    identities = [
        (row.family, row.qid)
        for block in selection.BLOCK_ORDER
        for row in selected[block]
    ]
    assert len(identities) == len(set(identities)) == selection.SELECTED_TOTAL


def test_action_projection_and_hash_match_frozen_semantic_contract_exactly() -> None:
    source = _synthetic_source()
    records = selection.validate_source_payload(source.value, contract=source.contract)
    selected = selection.select_private_blocks(records, secret=SECRET)
    actions, qrels = selection.build_private_packs(
        selected, secret=SECRET, task_objectives=OBJECTIVES
    )

    for block in selection.BLOCK_ORDER:
        for action_item, qrel_item, source_row in zip(
            actions[block]["items"],
            qrels[block]["items"],
            selected[block],
            strict=True,
        ):
            hipporag_input = action_item["hipporag_input"]
            projections = tuple(
                semantic_contract.project_candidate_text(
                    raw_text, candidate_ordinal=candidate_ordinal
                )
                for candidate_ordinal, raw_text in enumerate(
                    source_row.candidate_texts
                )
            )
            expected_documents = [
                {"ordinal": row.ordinal, "text": row.projection_text}
                for row in projections
            ]
            assert hipporag_input["documents"] == expected_documents
            assert all(
                document["text"] == projection.projection_text
                and document["text"] != raw_text
                and raw_text not in json.dumps(action_item, sort_keys=True)
                for document, projection, raw_text in zip(
                    hipporag_input["documents"],
                    projections,
                    source_row.candidate_texts,
                    strict=True,
                )
            )
            expected_hash = semantic_contract.common_projection_sha256(
                objective=OBJECTIVES[source_row.family],
                query=source_row.query,
                candidates=projections,
            )
            assert action_item["common_projection_sha256"] == expected_hash
            assert hipporag_input["common_projection_sha256"] == expected_hash
            assert [
                row["candidate_ordinal"] for row in qrel_item["qrel_values"]
            ] == list(range(len(source_row.candidate_ids)))
            assert [row["value"] for row in qrel_item["qrel_values"]] == list(
                source_row.qrel_values
            )


def test_commitment_precedes_parse_and_packs_are_private_and_label_separated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic_source()
    qualification_path, source_path, qualification_sha = _write_fixture(
        tmp_path, source
    )
    root = tmp_path / "ordered-acquisition"
    random_calls: list[int] = []
    original_parse = selection.parse_source_bytes

    def random_bytes(size: int) -> bytes:
        random_calls.append(size)
        return SECRET

    def observed_parse(raw: bytes, *, contract: selection.SourceContract):
        assert (root / selection.SECRET_FILENAME).read_bytes() == SECRET
        marker = _load(root / selection.COMMITMENT_FILENAME)
        assert marker["status"] == "one_32_byte_secret_committed_before_source_parse"
        assert marker["selection_secret_commitment_sha256"] == hashlib.sha256(
            SECRET
        ).hexdigest()
        return original_parse(raw, contract=contract)

    monkeypatch.setattr(selection, "parse_source_bytes", observed_parse)
    receipt = selection.acquire_once(
        source_path=source_path,
        qualification_path=qualification_path,
        output_root=root,
        contract=source.contract,
        expected_qualification_self_sha256=qualification_sha,
        task_objectives=OBJECTIVES,
        random_bytes=random_bytes,
    )
    assert random_calls == [32]
    assert receipt["selection_contract"]["selected_total"] == 120

    public_raw = (root / selection.PUBLIC_RECEIPT_FILENAME).read_text(
        encoding="ascii"
    )
    commitment_raw = (root / selection.COMMITMENT_FILENAME).read_text(
        encoding="ascii"
    )
    for family in selection.FAMILIES:
        forbidden = (
            f"PRIVATE_QID::{family}::000",
            f"PRIVATE_CID::{family}::00",
            f"PRIVATE_QUERY_TEXT::{family}::000",
            f"PRIVATE_DOCUMENT_TEXT::{family}::00",
            OBJECTIVES[family],
        )
        assert all(token not in public_raw for token in forbidden)
        assert all(token not in commitment_raw for token in forbidden)

    assert _mode(root / selection.SECRET_FILENAME) == 0o600
    assert _mode(root / selection.COMMITMENT_FILENAME) == 0o644
    assert _mode(root / selection.PUBLIC_RECEIPT_FILENAME) == 0o644
    assert (root / selection.SECRET_FILENAME).stat().st_size == 32
    for block in selection.BLOCK_ORDER:
        action_path = root / selection.ACTION_PACK_FILENAMES[block]
        qrel_path = root / selection.QREL_PACK_FILENAMES[block]
        assert _mode(action_path) == _mode(qrel_path) == 0o600
        assert not (root / f".{action_path.name}.part").exists()
        assert not (root / f".{qrel_path.name}.part").exists()
        action = _load(action_path)
        qrels = _load(qrel_path)
        assert action["numeric_qrel_value_included"] is False
        assert qrels["numeric_qrel_values_sealed_separately"] is True
        assert qrels["action_pack_sha256"] == action["action_pack_sha256"]
        assert len(action["items"]) == len(qrels["items"]) == 30
        assert [item["work_id"] for item in action["items"]] == [
            item["work_id"] for item in qrels["items"]
        ]
        for action_item, qrel_item in zip(
            action["items"], qrels["items"], strict=True
        ):
            assert set(action_item) == {
                "schema",
                "block_ordinal",
                "work_id",
                "candidate_count",
                "common_projection_sha256",
                "hipporag_input",
            }
            hipporag_input = action_item["hipporag_input"]
            assert set(hipporag_input) == {
                "schema",
                "work_id",
                "objective",
                "query",
                "documents",
                "common_projection_sha256",
            }
            assert hipporag_input["schema"] == (
                "birco_official_hipporag_candidate_retrieval_v1_input"
            )
            assert hipporag_input["work_id"] == action_item["work_id"]
            assert hipporag_input["common_projection_sha256"] == (
                action_item["common_projection_sha256"]
            )
            assert all(
                set(document) == {"ordinal", "text"}
                for document in hipporag_input["documents"]
            )
            expected_projection = selection.stable_hash(
                {
                    "documents": hipporag_input["documents"],
                    "objective": hipporag_input["objective"],
                    "query": hipporag_input["query"],
                }
            )
            assert action_item["common_projection_sha256"] == expected_projection
            assert "qrel_values" not in action_item
            assert "family" not in action_item
            assert len(qrel_item["qrel_values"]) == action_item["candidate_count"]
            assert all(
                set(row) == {"candidate_ordinal", "value"}
                for row in qrel_item["qrel_values"]
            )


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_duplicate_and_nonfinite_json_are_rejected(constant: bytes) -> None:
    duplicate = b'{"doris-mae":{},"doris-mae":{}}'
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="duplicate"):
        selection.parse_source_bytes(duplicate, contract=_synthetic_source().contract)
    nonfinite = b'{"doris-mae":' + constant + b"}"
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="non-finite"):
        selection.parse_source_bytes(nonfinite, contract=_synthetic_source().contract)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_python_qrel_values_are_rejected(score: float) -> None:
    source = _synthetic_source()
    value = deepcopy(source.value)
    qid = next(iter(value["doris-mae"]["qrel"]))
    cid = next(iter(value["doris-mae"]["qrel"][qid]))
    value["doris-mae"]["qrel"][qid][cid] = score
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="qrel value"):
        selection.validate_source_payload(value, contract=source.contract)


def test_full_candidate_membership_and_corpus_joins_are_enforced() -> None:
    source = _synthetic_source()

    absent_join = deepcopy(source.value)
    del absent_join["wtb"]["corpus"]["PRIVATE_CID::wtb::00"]
    with pytest.raises(selection.BircoP1PrivateSelectionError):
        selection.validate_source_payload(absent_join, contract=source.contract)

    missing_qrel_row = deepcopy(source.value)
    missing_qrel_row["clinical-trial"]["qrel"].pop(
        "PRIVATE_QID::clinical-trial::000"
    )
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="identity sets"):
        selection.validate_source_payload(missing_qrel_row, contract=source.contract)

    unused_corpus = deepcopy(source.value)
    unused_corpus["wtb"]["corpus"]["PRIVATE_UNUSED_CID::wtb"] = (
        "PRIVATE_UNUSED_DOCUMENT::wtb"
    )
    contracts = list(source.contract.families)
    contracts[-1] = selection.FamilyContract("wtb", 40, 11, (0.0, 1.0))
    expanded_contract = selection.SourceContract(
        source_size_bytes=source.contract.source_size_bytes,
        source_md5=source.contract.source_md5,
        source_sha256=source.contract.source_sha256,
        families=tuple(contracts),
        minimum_pool_size=10,
    )
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="complete family corpus"):
        selection.validate_source_payload(unused_corpus, contract=expanded_contract)


def test_one_shot_root_prevents_replay_and_second_random_call(tmp_path: Path) -> None:
    source = _synthetic_source()
    qualification_path, source_path, qualification_sha = _write_fixture(
        tmp_path, source
    )
    root = tmp_path / "one-shot"
    calls: list[int] = []

    def random_bytes(size: int) -> bytes:
        calls.append(size)
        return SECRET

    arguments = {
        "source_path": source_path,
        "qualification_path": qualification_path,
        "output_root": root,
        "contract": source.contract,
        "expected_qualification_self_sha256": qualification_sha,
        "task_objectives": OBJECTIVES,
        "random_bytes": random_bytes,
    }
    selection.acquire_once(**arguments)
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="replay"):
        selection.acquire_once(**arguments)
    assert calls == [32]


def test_exact_source_identity_failure_is_terminal_after_commitment(tmp_path: Path) -> None:
    source = _synthetic_source()
    qualification_path, source_path, qualification_sha = _write_fixture(
        tmp_path, source
    )
    source_path.write_bytes(source.raw + b" ")
    source_path.chmod(0o600)
    root = tmp_path / "source-drift"
    calls: list[int] = []

    def random_bytes(size: int) -> bytes:
        calls.append(size)
        return SECRET

    with pytest.raises(selection.BircoP1PrivateSelectionError, match="size"):
        selection.acquire_once(
            source_path=source_path,
            qualification_path=qualification_path,
            output_root=root,
            contract=source.contract,
            expected_qualification_self_sha256=qualification_sha,
            task_objectives=OBJECTIVES,
            random_bytes=random_bytes,
        )
    assert calls == [32]
    assert (root / selection.SECRET_FILENAME).read_bytes() == SECRET
    assert (root / selection.COMMITMENT_FILENAME).exists()
    assert (root / selection.FAILURE_FILENAME).exists()
    failure_raw = (root / selection.FAILURE_FILENAME).read_text(encoding="ascii")
    assert "PRIVATE_QID" not in failure_raw
    assert "PRIVATE_DOCUMENT_TEXT" not in failure_raw


def test_qualification_drift_fails_before_attempt_or_randomness(tmp_path: Path) -> None:
    source = _synthetic_source()
    qualification_path, source_path, qualification_sha = _write_fixture(
        tmp_path, source
    )
    qualification = _load(qualification_path)
    qualification["qualified"] = False
    qualification_path.write_bytes(_canonical(qualification))
    root = tmp_path / "qualification-drift"
    calls: list[int] = []
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="self-hash"):
        selection.acquire_once(
            source_path=source_path,
            qualification_path=qualification_path,
            output_root=root,
            contract=source.contract,
            expected_qualification_self_sha256=qualification_sha,
            task_objectives=OBJECTIVES,
            random_bytes=lambda size: calls.append(size) or SECRET,
        )
    assert calls == []
    assert not root.exists()


def test_block_qrels_require_exact_authorization_and_open_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _receipt, root, _source, _qualification, _source_path, _sha = _acquire(
        tmp_path
    )
    nonexistent = tmp_path / "does-not-exist.authorization.json"
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="permanently"):
        selection.open_block_qrels(
            output_root=root,
            block="F_search",
            authorization_path=nonexistent,
            expected_authorization_sha256="0" * 64,
        )
    assert not (root / selection.QREL_OPEN_MARKER_FILENAMES["F_search"]).exists()

    with pytest.raises(selection.BircoP1PrivateSelectionError, match="promotion"):
        selection.write_block_open_authorization(
            tmp_path / "m.invalid.authorization.json",
            output_root=root,
            block="M_search",
            action_archive_sha256s=["1" * 64],
        )

    authorization_path = tmp_path / "A_form.authorization.json"
    authorization = selection.write_block_open_authorization(
        authorization_path,
        output_root=root,
        block="A_form",
        action_archive_sha256s=["2" * 64, "1" * 64],
    )
    assert _mode(authorization_path) == 0o600
    assert authorization["action_archive_sha256s"] == ["1" * 64, "2" * 64]

    original_reader = selection._read_bound_private_pack
    reads: list[str] = []

    def observed_reader(*args: Any, **kwargs: Any):
        reads.append(str(kwargs.get("label")))
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(selection, "_read_bound_private_pack", observed_reader)
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="authorization"):
        selection.open_block_qrels(
            output_root=root,
            block="A_form",
            authorization_path=authorization_path,
            expected_authorization_sha256="f" * 64,
        )
    assert reads == []
    assert not (root / selection.QREL_OPEN_MARKER_FILENAMES["A_form"]).exists()

    opened = selection.open_block_qrels(
        output_root=root,
        block="A_form",
        authorization_path=authorization_path,
        expected_authorization_sha256=authorization["authorization_sha256"],
    )
    assert reads == ["A_form sealed qrel pack"]
    assert opened["block"] == "A_form"
    assert len(opened["items"]) == 30
    assert _mode(root / selection.QREL_OPEN_MARKER_FILENAMES["A_form"]) == 0o600

    with pytest.raises(selection.BircoP1PrivateSelectionError, match="already exists"):
        selection.open_block_qrels(
            output_root=root,
            block="A_form",
            authorization_path=authorization_path,
            expected_authorization_sha256=authorization["authorization_sha256"],
        )


def test_task_objectives_manifest_has_exact_self_hashed_readme_binding(
    tmp_path: Path,
) -> None:
    body = {
        "schema": selection.TASK_OBJECTIVES_MANIFEST_SCHEMA,
        "repository_commit": selection.BIRCO_REPOSITORY_COMMIT,
        "readme_sha256": "a" * 64,
        "objectives": OBJECTIVES,
    }
    manifest = selection.self_hashed(body, "self_sha256")
    path = tmp_path / "objectives.json"
    path.write_bytes(_canonical(manifest))
    assert selection.load_task_objectives_manifest(path) == OBJECTIVES

    invalid_bodies = {
        "extra": {**body, "mutable_note": "not allowed"},
        "commit": {**body, "repository_commit": "0" * 40},
        "readme": {**body, "readme_sha256": "not-a-sha256"},
        "families": {
            **body,
            "objectives": {
                family: objective
                for family, objective in OBJECTIVES.items()
                if family != "wtb"
            },
        },
        "objective_bound": {
            **body,
            "objectives": {
                **OBJECTIVES,
                "wtb": "x" * (selection.MAX_OBJECTIVE_CHARACTERS + 1),
            },
        },
    }
    for name, invalid_body in invalid_bodies.items():
        invalid = selection.self_hashed(invalid_body, "self_sha256")
        invalid_path = tmp_path / f"objectives.{name}.json"
        invalid_path.write_bytes(_canonical(invalid))
        with pytest.raises(selection.BircoP1PrivateSelectionError):
            selection.load_task_objectives_manifest(invalid_path)

    tampered = deepcopy(manifest)
    tampered["objectives"]["wtb"] = "tampered without rehash"
    tampered_path = tmp_path / "objectives.tampered.json"
    tampered_path.write_bytes(_canonical(tampered))
    with pytest.raises(selection.BircoP1PrivateSelectionError, match="self-hash"):
        selection.load_task_objectives_manifest(tampered_path)


def test_committed_qualification_result_self_hash_is_bound_without_source_parse() -> None:
    project = Path(__file__).resolve().parents[1]
    result = selection.verify_qualification_result(
        project / selection.QUALIFICATION_RELATIVE,
        expected_self_sha256=selection.QUALIFICATION_SELF_SHA256,
        contract=selection.FORMAL_CONTRACT,
    )
    assert result["self_sha256"] == selection.QUALIFICATION_SELF_SHA256
    assert result["source_identity"]["sha256"] == selection.SOURCE_SHA256
