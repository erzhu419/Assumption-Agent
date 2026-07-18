from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import unicodedata

import pytest

from assumption_agent.benchmarks import hover_direct_acquisition_v1 as acq


def _synthetic_sources() -> tuple[list[dict[str, object]], acq.DocumentCatalog]:
    documents = acq.build_document_catalog(
        [
            {
                "rowid": rowid,
                "id": unicodedata.normalize("NFD", f"Café document {rowid}"),
                "text": f"Unique synthetic body {rowid}",
            }
            for rowid in range(1, 1001)
        ]
    )
    payload: list[dict[str, object]] = []
    candidate = 0
    for hop in (2, 3, 4):
        for within_hop in range(60):
            first_rowid = candidate * 4 + 1
            support_rowids = range(first_rowid, first_rowid + hop)
            payload.append(
                {
                    "uid": f"uid-{hop}-{within_hop}",
                    "hpqa_id": "shared-cross-hop-group" if within_hop == 0 else f"group-{hop}-{within_hop}",
                    "claim": f"Synthetic claim {hop}-{within_hop}",
                    "num_hops": hop,
                    "supporting_facts": [
                        [f"Café document {rowid}", 0] for rowid in support_rowids
                    ],
                    "label": "SUPPORTED",
                }
            )
            candidate += 1
    for offset, claim in enumerate(("ＳＡＭＥ   claim", "same claim")):
        first_rowid = (candidate + offset) * 4 + 1
        payload.append(
            {
                "uid": f"collision-uid-{offset}",
                "hpqa_id": f"collision-group-{offset}",
                "claim": claim,
                "num_hops": 2,
                "supporting_facts": [
                    [f"Café document {first_rowid}", 0],
                    [f"Café document {first_rowid + 1}", 1],
                ],
                "label": "SUPPORTED",
            }
        )
    return payload, documents


def _qualified(
    payload: list[dict[str, object]], documents: acq.DocumentCatalog
) -> tuple[tuple[acq.EligibleRecord, ...], dict[str, object], acq.QualificationBinding]:
    eligible, stats = acq.parse_train_payload(payload, documents=documents)
    qualification = acq.synthetic_qualification_binding(
        source_stats=stats, documents=documents
    )
    return eligible, stats, qualification


def test_synthetic_selection_fixed_corpus_and_label_isolation() -> None:
    payload, documents = _synthetic_sources()
    eligible, stats, qualification = _qualified(payload, documents)
    assert len(eligible) == 180
    assert stats["normalized_claim_collision_group_count"] == 1
    assert stats["normalized_claim_collision_member_count"] == 2
    assert stats["structural_exclusion_counts"] == {}

    secret = bytes(range(32))
    blocks, selection = acq.select_private_blocks(
        eligible,
        source_stats=stats,
        qualification=qualification,
        secret=secret,
    )
    assert selection["selected_block_counts"] == acq.BLOCK_COUNTS
    flattened = [row for block in acq.BLOCK_ORDER for row in blocks[block]]
    assert len(flattened) == 144
    assert len({row.record.hpqa_id_sha256 for row in flattened}) == 144
    for block in acq.BLOCK_ORDER:
        assert Counter(row.hop_stratum for row in blocks[block]) == Counter(
            {hop: acq.HOP_QUOTAS[block] for hop in acq.HOP_STRATA}
        )

    corpus_rows, article_ids, corpus_stats = acq.build_fixed_corpus(
        blocks=blocks,
        documents=documents,
        qualification=qualification,
        secret=secret,
    )
    assert len(corpus_rows) == acq.CORPUS_SIZE
    assert corpus_stats["all_selected_gold_included"] is True
    assert len({row.exact_id + "\n\n" + row.exact_text for row in corpus_rows}) == 609

    corpus, views, labels, materialization = acq.materialize_private_payloads(
        blocks=blocks,
        corpus_rows=corpus_rows,
        article_id_by_rowid=article_ids,
    )
    assert set(labels) == {"A_form", "A_hold", "M_search"}
    assert materialization["F_search_utility_label_pack_created"] is False
    assert all(set(article) == {"article_id", "title", "body"} for article in corpus["articles"])
    for block, view in views.items():
        assert len(view["items"]) == acq.BLOCK_COUNTS[block]
        assert all(set(item) == {"schema", "block", "ordinal", "claim"} for item in view["items"])
        assert all("hop_stratum" not in item and "gold_article_ids" not in item for item in view["items"])
    for block, label_pack in labels.items():
        for item in label_pack["items"]:
            assert len(item["gold_article_ids"]) == int(item["hop_stratum"][0])
            assert all(0 <= article_id < 609 for article_id in item["gold_article_ids"])


def test_selection_and_corpus_are_deterministic_under_source_permutation() -> None:
    payload, documents = _synthetic_sources()
    outputs = []
    for source in (payload, list(reversed(payload))):
        eligible, stats, qualification = _qualified(source, documents)
        blocks, _selection = acq.select_private_blocks(
            eligible,
            source_stats=stats,
            qualification=qualification,
            secret=b"D" * 32,
        )
        corpus, article_ids, _stats = acq.build_fixed_corpus(
            blocks=blocks,
            documents=documents,
            qualification=qualification,
            secret=b"D" * 32,
        )
        outputs.append(
            (
                {
                    block: [row.record.identity_commitment_sha256 for row in blocks[block]]
                    for block in acq.BLOCK_ORDER
                },
                [row.identity_commitment_sha256 for row in corpus],
                dict(article_ids),
            )
        )
    assert outputs[0] == outputs[1]


def test_one_shot_persistence_loaders_modes_and_tamper_detection(tmp_path: Path) -> None:
    payload, documents = _synthetic_sources()
    _eligible, stats, qualification = _qualified(payload, documents)
    calls: list[int] = []

    def fixed_random(length: int) -> bytes:
        calls.append(length)
        return b"S" * length

    receipt = acq.execute_acquisition_once(
        train_payload=payload,
        documents=documents,
        qualification=qualification,
        paths=acq.default_acquisition_paths(tmp_path),
        source_bindings={"synthetic": True, "source_stats_sha256": acq.stable_hash(stats)},
        random_bytes=fixed_random,
    )
    assert calls == [32]
    assert receipt["status"] == "private_four_block_pack_formed"
    paths = acq.default_acquisition_paths(tmp_path)
    for private_path in (
        paths.marker,
        paths.secret,
        paths.corpus_view,
        *paths.block_views.values(),
        *paths.block_labels.values(),
    ):
        assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert not paths.block_labels.get("F_search")
    assert stat.S_IMODE(paths.public_receipt.stat().st_mode) == 0o644

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", acq.PUBLIC_RECEIPT_RELATIVE.as_posix()],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=synthetic-test",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-q",
            "-m",
            "commit aggregate receipt",
        ],
        check=True,
    )

    loaded, bindings = acq.load_committed_acquisition_receipt(tmp_path)
    assert loaded["acquisition_sha256"] == receipt["acquisition_sha256"]
    assert bindings["receipt_file_sha256"]
    assert bindings["receipt_git_head"]
    assert bindings["receipt_git_blob_sha1"]
    with pytest.raises(
        acq.HoVerAcquisitionError, match="formal acquisition source binding"
    ):
        acq.load_formal_committed_acquisition_receipt(tmp_path)
    assert len(acq.load_corpus_view(project=tmp_path)["articles"]) == 609
    for block in acq.BLOCK_ORDER:
        assert len(acq.load_block_view(project=tmp_path, expected_block=block)["items"]) == acq.BLOCK_COUNTS[block]
    with pytest.raises(acq.HoVerAcquisitionError, match="does not exist"):
        acq.load_block_labels(project=tmp_path, expected_block="F_search")

    with pytest.raises(acq.HoVerAcquisitionError, match="replay forbidden"):
        acq.execute_acquisition_once(
            train_payload=payload,
            documents=documents,
            qualification=qualification,
            paths=paths,
            source_bindings={"synthetic": True},
            random_bytes=lambda length: b"X" * length,
        )
    assert calls == [32]

    receipt_raw = paths.public_receipt.read_bytes()
    paths.public_receipt.write_bytes(receipt_raw + b"\n")
    with pytest.raises(acq.HoVerAcquisitionError, match="stable Git HEAD"):
        acq.load_committed_acquisition_receipt(tmp_path)
    paths.public_receipt.write_bytes(receipt_raw)

    raw = paths.block_views["A_form"].read_bytes()
    paths.block_views["A_form"].write_bytes(raw.replace(b"Synthetic claim", b"Synthetic claiM", 1))
    with pytest.raises(acq.HoVerAcquisitionError, match="file binding drifted"):
        acq.load_block_view(project=tmp_path, expected_block="A_form")


def test_strict_json_committed_qualification_and_synthetic_sqlite(tmp_path: Path) -> None:
    with pytest.raises(acq.HoVerAcquisitionError, match="duplicate object key"):
        acq.strict_json_loads(b'{"a":1,"a":2}', label="duplicate")
    with pytest.raises(acq.HoVerAcquisitionError, match="non-finite"):
        acq.strict_json_loads(b'{"a":NaN}', label="nan")

    project = Path(__file__).parents[1]
    qualification = json.loads(
        (project / "manifests/hover_source_qualification_v1.json").read_text()
    )
    binding = acq.validate_qualification_manifest(qualification)
    assert binding.eligible_record_count == 17_905
    assert binding.sqlite_document_row_count == 5_233_329

    database = tmp_path / "synthetic.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE documents (id TEXT, text TEXT)")
    connection.executemany(
        "INSERT INTO documents(id, text) VALUES (?, ?)",
        [
            (unicodedata.normalize("NFD", f"Café {rowid}"), f"body {rowid}")
            for rowid in range(1, 621)
        ],
    )
    connection.commit()
    connection.close()
    os.chmod(database, 0o600)
    with acq.ImmutableSQLiteDocumentResolver(
        path=database,
        row_count=620,
        maximum_rowid=620,
        binding_sha256="a" * 64,
    ) as resolver:
        resolved = resolver.resolve_exact_nfd_id("Café 1")
        assert len(resolved) == 1
        assert resolved[0].rowid == 1
        assert resolver.fetch_rowid(620).exact_text == "body 620"
