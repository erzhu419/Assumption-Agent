from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path
import zipfile

import pytest

from replication_runtime.financial_semantic_v2 import oracle_pandas
from replication_runtime.financial_semantic_v2 import oracle_streaming
from replication_runtime.financial_semantic_v2 import pack as period_pack


MANAGER_COUNT = 32
ISSUER_COUNT = 20


def _write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_period(root: Path, *, current: bool) -> None:
    report_date = "31-DEC-2025" if current else "30-SEP-2025"
    prefix = "C" if current else "P"
    cover_rows: list[list[object]] = []
    info_rows: list[list[object]] = []
    for manager_index in range(MANAGER_COUNT):
        accession = f"{prefix}{manager_index:05d}"
        manager = f"Period Fund {manager_index:02d} LLC"
        cover_rows.append(
            [accession, report_date, "13F HOLDINGS REPORT", manager]
        )
        for issuer_index in range(ISSUER_COUNT):
            cusip = f"{100_000_000 + issuer_index:09d}"
            previous_value = 10_000 + manager_index * 100 + issuer_index
            value = (
                previous_value + (issuer_index + 1) * 1_000 + manager_index
                if current
                else previous_value
            )
            info_rows.append(
                [
                    accession,
                    f"Issuer Corporation {issuer_index:02d}",
                    "COM",
                    cusip,
                    value,
                ]
            )
        # It contributes to AUM but is intentionally not a stock row and is
        # held by only one manager, so it cannot enter the issuer pool.
        info_rows.append(
            [
                accession,
                f"Private Note {manager_index:02d}",
                "PUT",
                f"{900_000_000 + manager_index:09d}",
                777 + manager_index,
            ]
        )
    cover_rows.extend(
        [
            ["OLD000", "30-SEP-2025", "13F HOLDINGS REPORT", "Old Fund"],
            ["NOTICE0", report_date, "13F NOTICE", "Notice Fund"],
        ]
    )
    info_rows.extend(
        [
            ["OLD000", "Ignored Old Issuer", "COM", "800000001", 999_999_999],
            ["NOTICE0", "Ignored Notice Issuer", "COM", "800000002", 999_999_999],
        ]
    )
    if current:
        # Row-order invariance is part of both oracle contracts.
        cover_rows.reverse()
        info_rows.reverse()
    _write_tsv(
        root / "COVERPAGE.tsv",
        [
            "ACCESSION_NUMBER",
            "REPORTCALENDARORQUARTER",
            "REPORTTYPE",
            "FILINGMANAGER_NAME",
        ],
        cover_rows,
    )
    _write_tsv(
        root / "INFOTABLE.tsv",
        [
            "ACCESSION_NUMBER",
            "NAMEOFISSUER",
            "TITLEOFCLASS",
            "CUSIP",
            "VALUE",
        ],
        info_rows,
    )


def _zip_period(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.iterdir()):
            archive.write(path, arcname=f"official-sec-period/{path.name}")


@pytest.fixture()
def sources(tmp_path: Path) -> dict[str, Path]:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_period(previous, current=False)
    _write_period(current, current=True)
    previous_zip = tmp_path / "previous.zip"
    current_zip = tmp_path / "current.zip"
    _zip_period(previous, previous_zip)
    _zip_period(current, current_zip)
    return {
        "previous": previous,
        "current": current,
        "previous_zip": previous_zip,
        "current_zip": current_zip,
    }


def _build(sources: dict[str, Path], *, seed: str = "period-out-test-seed") -> dict:
    return period_pack.build_public_pack(
        previous_source=sources["previous_zip"],
        current_source=sources["current_zip"],
        previous_period_label="2025 Q3",
        current_period_label="2025 Q4",
        preregistration_seed=seed,
        previous_container_root="/root/2025-q2",
        current_container_root="/root/2025-q3",
    )


def _rehash(payload: dict, field: str) -> None:
    body = dict(payload)
    body.pop(field, None)
    payload[field] = period_pack.payload_hash(body)


def test_pack_is_deterministic_and_has_frozen_4fold_x2_plus_4sealed_layout(
    sources: dict[str, Path],
) -> None:
    first = _build(sources)
    second = _build(sources)
    assert first == second
    assert period_pack.verify_public_pack(first) == first
    assert first["ground_truth_persisted"] is False
    assert first["candidate_imports"] == 0
    assert first["model_calls"] == 0
    assert first["network_calls"] == 0
    assert first["snapshot_report_dates"] == {
        "previous": "2025-09-30",
        "current": "2025-12-31",
    }
    assert first["container_roots"] == {
        "previous": "/root/2025-q2",
        "current": "/root/2025-q3",
    }
    assert first["sources"]["previous"]["period_label"] == "2025 Q3"
    assert first["sources"]["current"]["period_label"] == "2025 Q4"

    measurement = period_pack.partition_items(first, "measurement")
    sealed = period_pack.partition_items(first, "sealed")
    assert len(measurement) == 8
    assert len(sealed) == 4
    assert {item["fold"] for item in measurement} == {0, 1, 2, 3}
    assert all(
        sum(item["fold"] == fold for item in measurement) == 2
        for fold in range(4)
    )
    assert [item["template"] for item in measurement] == [
        "four_question_v1",
        "three_question_v1",
    ] * 4
    assert all("answers" not in item and "expected_output" not in item for item in first["items"])

    changed_seed = _build(sources, seed="another-preregistered-seed")
    assert changed_seed["selection_seed"] != first["selection_seed"]
    assert changed_seed["pack_hash"] != first["pack_hash"]

    reversed_dates = copy.deepcopy(first)
    reversed_dates["snapshot_report_dates"] = {
        "previous": "2025-12-31",
        "current": "2025-09-30",
    }
    _rehash(reversed_dates, "pack_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="period order"):
        period_pack.verify_public_pack(reversed_dates)

    # ZIP formation and extracted evaluation bind the same table bytes.
    for role in ("previous", "current"):
        extracted = period_pack.Sec13FSource.open(sources[role])
        assert extracted.source_fingerprint == first["sources"][role][
            "source_fingerprint"
        ]


def test_measurement_view_redacts_every_sealed_query_entity_and_instruction(
    sources: dict[str, Path], tmp_path: Path
) -> None:
    private = _build(sources)
    view = period_pack.build_measurement_view(private)
    assert period_pack.verify_measurement_view(view, private_pack=private) == view
    assert len(view["measurement_items"]) == 8
    assert len(view["sealed_item_commitments"]) == 4
    assert "selection_seed" not in view
    assert private["selection_seed"] not in json.dumps(view, sort_keys=True)

    serialized = json.dumps(view, ensure_ascii=False, sort_keys=True)
    sealed_items = period_pack.partition_items(private, "sealed")
    for item, commitment in zip(
        sealed_items, view["sealed_item_commitments"]
    ):
        assert item["item_id"] == commitment["item_id"]
        assert set(commitment) == {
            "item_id",
            "template",
            "fold",
            "instruction_sha256",
            "query_commitment_hash",
            "full_item_commitment_hash",
        }
        assert item["instruction"] not in serialized
        for entity_key in ("aum_manager", "increase_manager", "issuer"):
            assert item["query"][entity_key] not in serialized

    private_path = tmp_path / "private.pack.json"
    view_path = tmp_path / "measurement.view.json"
    period_pack.write_json(private_path, private)
    assert period_pack.main(
        [
            "measurement-view",
            "--pack",
            str(private_path),
            "--output",
            str(view_path),
        ]
    ) == 0
    assert period_pack.read_json(view_path) == view


def test_pandas_and_stdlib_oracles_agree_with_separate_measurement_and_sealed_gold(
    sources: dict[str, Path],
) -> None:
    private = _build(sources)
    consensus: dict[str, dict] = {}
    for partition, expected_count in (("measurement", 8), ("sealed", 4)):
        pandas_output = oracle_pandas.evaluate_partition(
            pack=private,
            previous_source=sources["previous"],
            current_source=sources["current"],
            partition=partition,
        )
        streaming_output = oracle_streaming.evaluate_partition(
            pack=private,
            previous_source=sources["previous"],
            current_source=sources["current"],
            partition=partition,
        )
        assert [row["answers_hash"] for row in pandas_output["items"]] == [
            row["answers_hash"] for row in streaming_output["items"]
        ]
        gold = period_pack.build_consensus_gold(
            pack=private,
            left=pandas_output,
            right=streaming_output,
            partition=partition,
        )
        assert period_pack.verify_consensus_gold(
            gold,
            pack=private,
            expected_partition=partition,
        ) == gold
        assert gold["item_count"] == expected_count
        assert gold["cross_oracle_agreement"] is True
        consensus[partition] = gold

    measurement_ids = {row["item_id"] for row in consensus["measurement"]["items"]}
    sealed_ids = {row["item_id"] for row in consensus["sealed"]["items"]}
    assert measurement_ids.isdisjoint(sealed_ids)
    assert not any(item_id in json.dumps(consensus["measurement"]) for item_id in sealed_ids)
    assert not any(item_id in json.dumps(consensus["sealed"]) for item_id in measurement_ids)

    # Every four-question item has exactly the 20 COM rows in the fixture;
    # its private-note PUT row contributes to AUM but not to stock count.
    private_by_id = {item["item_id"]: item for item in private["items"]}
    for row in consensus["measurement"]["items"] + consensus["sealed"]["items"]:
        if private_by_id[row["item_id"]]["template"] == "four_question_v1":
            assert row["answers"]["q2_answer"] == ISSUER_COUNT


def test_hash_closed_redaction_and_cross_oracle_disagreement_fail_closed(
    sources: dict[str, Path],
) -> None:
    private = _build(sources)
    tampered_pack = copy.deepcopy(private)
    tampered_pack["items"][0]["query"]["aum_manager"] = "Injected Manager"
    _rehash(tampered_pack, "pack_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="instruction"):
        period_pack.verify_public_pack(tampered_pack)

    hidden_pack = copy.deepcopy(private)
    hidden_pack["items"][0]["hidden_ground_truth"] = {"answer": 1}
    _rehash(hidden_pack, "pack_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="item fields"):
        period_pack.verify_public_pack(hidden_pack)

    view = period_pack.build_measurement_view(private)
    tampered_view = copy.deepcopy(view)
    tampered_view["sealed_item_commitments"][0]["query"] = {
        "issuer": "leaked"
    }
    _rehash(tampered_view, "measurement_view_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="extra fields"):
        period_pack.verify_measurement_view(tampered_view)

    hidden_view = copy.deepcopy(view)
    hidden_view["measurement_items"][0]["sealed_answer"] = "leaked"
    _rehash(hidden_view, "measurement_view_hash")
    with pytest.raises(
        period_pack.PeriodOutPackError,
        match="measurement item fields",
    ):
        period_pack.verify_measurement_view(hidden_view)

    left = oracle_pandas.evaluate_partition(
        pack=private,
        previous_source=sources["previous"],
        current_source=sources["current"],
        partition="measurement",
    )
    right = oracle_streaming.evaluate_partition(
        pack=private,
        previous_source=sources["previous"],
        current_source=sources["current"],
        partition="measurement",
    )
    forged = copy.deepcopy(left)
    forged["oracle_id"] = "renamed_copy_of_one_oracle"
    _rehash(forged, "oracle_output_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="policy"):
        period_pack.verify_oracle_output(
            forged,
            pack=private,
            expected_partition="measurement",
        )

    valid_gold = period_pack.build_consensus_gold(
        pack=private,
        left=left,
        right=right,
        partition="measurement",
    )
    leaked_gold = copy.deepcopy(valid_gold)
    leaked_gold["sealed_payload"] = {"answer": "forbidden"}
    _rehash(leaked_gold, "gold_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="fields drifted"):
        period_pack.verify_consensus_gold(
            leaked_gold,
            pack=private,
            expected_partition="measurement",
        )

    right = copy.deepcopy(right)
    right["items"][0]["answers"]["q1_answer"] += 1
    right["items"][0]["answers_hash"] = period_pack.payload_hash(
        right["items"][0]["answers"]
    )
    _rehash(right, "oracle_output_hash")
    with pytest.raises(period_pack.PeriodOutPackError, match="disagree"):
        period_pack.build_consensus_gold(
            pack=private,
            left=left,
            right=right,
            partition="measurement",
        )


def test_sec_zip_rejects_path_traversal_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../COVERPAGE.tsv", "header\n")
        handle.writestr("INFOTABLE.tsv", "header\n")

    with pytest.raises(period_pack.PeriodOutPackError, match="unsafe member"):
        period_pack.Sec13FSource.open(archive)


def test_oracle_modules_have_no_candidate_or_assumption_agent_imports() -> None:
    paths = [
        Path(period_pack.__file__),
        Path(oracle_pandas.__file__),
        Path(oracle_streaming.__file__),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name.startswith("assumption_agent") or name.startswith("candidates")
            for name in imported
        )
    streaming_tree = ast.parse(
        Path(oracle_streaming.__file__).read_text(encoding="utf-8")
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "pandas" for alias in node.names)
        for node in ast.walk(streaming_tree)
    )
