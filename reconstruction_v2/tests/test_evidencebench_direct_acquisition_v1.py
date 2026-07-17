from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks import evidencebench_direct_acquisition_v1 as e


def _sentences(paper_index: int, count: int = 48) -> list[str]:
    return [
        f"Synthetic paper {paper_index} sentence {sentence_index}."
        for sentence_index in range(count)
    ]


def _paper(
    paper_index: int,
    *,
    paper_id: str | None = None,
    sentence_count: int = 48,
) -> dict[str, object]:
    return {
        "paper_id": paper_id if paper_id is not None else f"SYNTH-{paper_index:03d}",
        "hypothesis": f"Synthetic hypothesis {paper_index}",
        "paper_as_candidate_pool": _sentences(paper_index, sentence_count),
        "aspect_list_ids": ["aspect-a", "aspect-b"],
        "aspect2sentence_indices": {
            "aspect-a": [0, 1, sentence_count - 1],
            "aspect-b": [2, sentence_count // 2],
        },
        # Known auxiliary README fields are deliberately ignored as features.
        "sentence_index2aspects": {},
        "evaluation": {"unused": True},
    }


def _payload() -> list[object]:
    return [_paper(index) for index in range(e.ROOT_RECORD_COUNT)]


def _write_source(path: Path, payload: object) -> Path:
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return path


def _paths(tmp_path: Path) -> e.OutputPaths:
    return e.OutputPaths(
        marker=tmp_path / "custody" / "attempt.marker",
        failure=tmp_path / "custody" / "failure.json",
        public_receipt=tmp_path / "public" / "receipt.json",
        private={
            block: (
                tmp_path / "private" / f"{block}.label-free.json",
                tmp_path / "private" / f"{block}.labels.json",
            )
            for block in e.BLOCK_ORDER
        },
    )


def _flatten(
    blocks: dict[str, tuple[e.EligibleItem, ...]],
) -> tuple[e.EligibleItem, ...]:
    return tuple(item for block in e.BLOCK_ORDER for item in blocks[block])


def test_source_identity_constants_and_missing_external_freeze_blocks_formal_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert e.SOURCE_REPOSITORY == "EvidenceBench/EvidenceBench"
    assert e.SOURCE_COMMIT == "bf1d9633c694381c7b016fd56ee9f95f48593cc3"
    assert e.SOURCE_REPOSITORY_PATH == "datasets/evidencebench_test_set.json"
    assert e.SOURCE_GIT_BLOB_SHA1 == "df380a1ba1359f9cea8bca2f2298dc9fd99e6513"
    assert e.SOURCE_BYTE_SIZE == 12_735_397
    assert e.SOURCE_COMMIT in e.SOURCE_RAW_URL
    assert e.SOURCE_REPOSITORY_PATH in e.SOURCE_RAW_URL
    assert e.IMPLEMENTATION_FREEZE_RELATIVE == (
        "manifests/evidencebench_implementation_freeze_v1.json"
    )
    source_touched = {"value": False}

    def forbidden_source_hash(_path: Path) -> e.SourceBinding:
        source_touched["value"] = True
        raise AssertionError("source was touched before the freeze existed")

    monkeypatch.setattr(e, "hash_source_file", forbidden_source_hash)
    with pytest.raises(e.EvidenceBenchAcquisitionError, match="unavailable"):
        e.verify_formal_protocol(
            project=tmp_path,
            source=tmp_path / "source.json",
            selection_secret=tmp_path / "secret.key",
        )
    assert source_touched["value"] is False


def test_public_exports_are_resolvable() -> None:
    assert e.__all__
    assert all(hasattr(e, name) for name in e.__all__)


def test_balanced_32_nodes_and_per_aspect_sentence_to_bucket_mapping() -> None:
    sentences = _sentences(7, 48)
    nodes = e.balanced_nodes(sentences)
    assert len(nodes) == 32
    assert [node.span_i for node in nodes] == list(range(32))
    assert nodes[0].start == 0 and nodes[0].end == 1
    assert nodes[1].start == 1 and nodes[1].end == 3
    assert nodes[-1].start == 46 and nodes[-1].end == 48
    assert max(node.end - node.start for node in nodes) == 2
    assert min(node.end - node.start for node in nodes) == 1
    assert all(node.identity_text == "\n".join(sentences[node.start : node.end]) for node in nodes)

    assert e.map_sentence_indices_to_nodes(
        [0, 1, 2, 47], nodes=nodes, sentence_count=48
    ) == (0, 1, 31)
    assert e.map_sentence_indices_to_nodes(
        [24, 24], nodes=nodes, sentence_count=48
    ) == (16,)
    with pytest.raises(e.LocalPaperError, match="aspect_sentence_index_bounds"):
        e.map_sentence_indices_to_nodes(
            [48], nodes=nodes, sentence_count=48
        )


def test_label_free_parser_does_not_read_gold_and_label_controller_keeps_all_aspects() -> None:
    class GoldGuard(dict[str, object]):
        gold_allowed = False

        def get(self, key, default=None):
            if key in {"aspect_list_ids", "aspect2sentence_indices"} and not self.gold_allowed:
                raise AssertionError("label-free parser touched gold")
            return super().get(key, default)

    record = GoldGuard(_paper(1))
    candidate = e._parse_label_free_paper(record, source_ordinal=0)
    assert candidate.hypothesis == "Synthetic hypothesis 1"
    assert len(candidate.nodes) == 32
    record.gold_allowed = True
    gold = e._parse_aspect_labels(record, candidate=candidate)
    assert gold == ((0, 1, 31), (1, 16))

    record["aspect_list_ids"] = ["aspect-a", "invalid-aspect"]
    record["aspect2sentence_indices"] = {
        "aspect-a": [0],
        "invalid-aspect": [],
    }
    with pytest.raises(e.LocalPaperError, match="aspect_sentence_list_schema"):
        e._parse_aspect_labels(record, candidate=candidate)


def test_exact_293_to_256_hmac_selection_is_deterministic_and_paper_disjoint() -> None:
    payload = _payload()
    blocks_a, stats_a = e.select_blocks_from_payload(payload, secret=bytes(range(32)))
    blocks_again, _ = e.select_blocks_from_payload(payload, secret=bytes(range(32)))
    blocks_b, _ = e.select_blocks_from_payload(payload, secret=b"b" * 32)

    selected_a = _flatten(blocks_a)
    selected_again = _flatten(blocks_again)
    selected_b = _flatten(blocks_b)
    assert [len(blocks_a[block]) for block in e.BLOCK_ORDER] == [64, 64, 64, 64]
    assert len(selected_a) == 256
    assert len({item.paper_commitment_sha256 for item in selected_a}) == 256
    assert [item.item_commitment_sha256 for item in selected_a] == [
        item.item_commitment_sha256 for item in selected_again
    ]
    assert [item.item_commitment_sha256 for item in selected_a] != [
        item.item_commitment_sha256 for item in selected_b
    ]
    assert stats_a["root_counts"]["declared_paper_records"] == 293
    assert stats_a["paper_counts"] == {
        "eligible": 293,
        "required": 256,
        "unused_eligible_after_selection": 37,
        "capacity_satisfied": True,
    }


def test_exposure_denylist_uses_only_id_or_explicit_metadata_not_citing_text() -> None:
    payload = _payload()
    payload[0] = _paper(0, paper_id=e.EXPOSED_PMCID)
    payload[1] = _paper(1, paper_id=e.EXPOSED_DOI.upper())
    payload[2] = _paper(2, paper_id=e.EXPOSED_URL)
    payload[3]["doi"] = e.EXPOSED_DOI
    citing = _paper(4)
    citing_sentences = citing["paper_as_candidate_pool"]
    assert isinstance(citing_sentences, list)
    citing_sentences[0] = f"This unrelated paper cites {e.EXPOSED_DOI}."
    payload[4] = citing

    blocks, stats = e.select_blocks_from_payload(payload, secret=b"x" * 32)
    assert len(_flatten(blocks)) == 256
    assert stats["exposure_counts"] == {
        "identifier_excluded_paper_components": 4
    }
    assert stats["paper_counts"]["eligible"] == 289
    # The only four exclusions are the explicit metadata identifiers; merely
    # citing the DOI in paper text did not add a fifth exposure exclusion.


def test_duplicate_and_schema_anomalies_exclude_whole_papers_without_runner_up() -> None:
    payload = _payload()
    payload[1]["paper_id"] = payload[0]["paper_id"]
    payload[2] = None
    payload[3]["paper_as_candidate_pool"] = _sentences(3, 47)
    payload[4]["aspect2sentence_indices"] = {"aspect-a": [0]}
    payload[5]["aspect2sentence_indices"] = {
        "aspect-a": [0],
        "aspect-b": [],
    }
    payload[6]["aspect2sentence_indices"] = {
        "aspect-a": [0],
        "aspect-b": [48],
    }
    payload[7]["paper_as_candidate_pool"] = list(
        payload[8]["paper_as_candidate_pool"]
    )

    blocks, stats = e.select_blocks_from_payload(payload, secret=b"z" * 32)
    reasons = stats["parser_reason_counts"]
    assert reasons["paper_not_object"] == 1
    assert reasons["sentence_count"] == 1
    assert reasons["aspect_map_key_mismatch"] == 1
    assert reasons["aspect_sentence_list_schema"] == 1
    assert reasons["aspect_sentence_index_bounds"] == 1
    assert stats["root_counts"]["multi_record_paper_components"] == 2
    assert stats["root_counts"]["paper_component_size_histogram"]["2"] == 2
    assert stats["paper_counts"]["eligible"] == 286
    assert len(_flatten(blocks)) == 256

    with pytest.raises(e.EvidenceBenchAcquisitionError, match="exactly 293"):
        e.select_blocks_from_payload(payload[:-1], secret=b"z" * 32)


def test_component_representative_is_label_blind_and_has_no_runner_up() -> None:
    payload = _payload()
    payload[1]["paper_id"] = payload[0]["paper_id"]
    secret = b"r" * 32
    rows = [
        (
            e._parse_label_free_paper(payload[index], source_ordinal=index),
            payload[index],
        )
        for index in (0, 1)
    ]
    commitment = e.component_commitment(rows)
    representative, representative_record = min(
        rows,
        key=lambda pair: (
            e._hmac_array(
                secret,
                [
                    "evidencebench_direct_v1",
                    "component_representative",
                    commitment,
                    pair[0].item_commitment_sha256,
                ],
            ),
            pair[0].item_commitment_sha256,
            pair[0].source_ordinal,
        ),
    )
    assert representative.source_ordinal in {0, 1}
    representative_record["aspect2sentence_indices"] = {
        "aspect-a": [0],
        "aspect-b": [],
    }
    blocks, stats = e.select_blocks_from_payload(payload, secret=secret)
    assert stats["root_counts"]["multi_record_paper_components"] == 1
    assert stats["parser_reason_counts"]["aspect_sentence_list_schema"] == 1
    # 293 records -> 292 components; the invalid selected representative burns
    # its component despite the other raw record having valid labels.
    assert stats["paper_counts"]["eligible"] == 291
    assert len(_flatten(blocks)) == 256


def test_marker_precedes_parse_no_replay_private_schema_and_public_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    source = _write_source(tmp_path / "synthetic-evidencebench.json", payload)
    binding = e.hash_source_file(source)
    paths = _paths(tmp_path)
    original_read = e._read_bound_source
    observed = {"read_count": 0}

    def checked_read(path: Path, initial: e.SourceBinding) -> bytes:
        observed["read_count"] += 1
        assert paths.marker.is_file()
        assert stat.S_IMODE(paths.marker.stat().st_mode) == 0o600
        marker = json.loads(paths.marker.read_text(encoding="utf-8"))
        assert marker["source_JSON_opened_or_parsed_before_marker"] is False
        return original_read(path, initial)

    monkeypatch.setattr(e, "_read_bound_source", checked_read)
    receipt = e.execute_acquisition_once(
        source_path=source,
        source_binding=binding,
        secret=b"m" * 32,
        protocol_bindings={"synthetic_fixture": True},
        paths=paths,
    )
    assert observed["read_count"] == 1
    assert receipt["status"] == "private_four_block_pack_formed"
    assert receipt["blocks"]["selected_item_count"] == 256
    assert receipt["blocks"]["global_paper_disjointness"] is True
    assert receipt["safety"]["performance_scores_computed"] == 0
    assert receipt["safety"]["model_calls"] == 0
    assert receipt["attempt"]["parser_or_model_worker_count"] == 0
    assert receipt["attempt"]["readonly_git_metadata_subprocess_count"] == 0

    all_commitments: list[str] = []
    for block in e.BLOCK_ORDER:
        view_path, label_path = paths.private[block]
        assert stat.S_IMODE(view_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(label_path.stat().st_mode) == 0o600
        views = json.loads(view_path.read_text(encoding="utf-8"))
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        assert views["schema"] == e.LABEL_FREE_BLOCK_SCHEMA
        assert labels["schema"] == e.LABEL_BLOCK_SCHEMA
        assert views["count"] == labels["count"] == 64
        for ordinal, (view, label) in enumerate(
            zip(views["rows"], labels["rows"], strict=True)
        ):
            assert view["ordinal"] == label["ordinal"] == ordinal
            assert view["item_commitment_sha256"] == label["item_commitment_sha256"]
            assert len(view["nodes"]) == 32
            assert "aspect_list_ids" not in label
            assert "aspect2sentence_indices" not in label
            assert all(label["gold_aspect_node_indices"])
            assert all(
                bucket_set == sorted(set(bucket_set))
                and all(0 <= value < 32 for value in bucket_set)
                for bucket_set in label["gold_aspect_node_indices"]
            )
            all_commitments.append(view["paper_commitment_sha256"])
    assert len(all_commitments) == len(set(all_commitments)) == 256

    public_raw = paths.public_receipt.read_text(encoding="utf-8")
    assert "Synthetic hypothesis" not in public_raw
    assert "Synthetic paper" not in public_raw
    assert "aspect-a" not in public_raw
    public = json.loads(public_raw)
    declared = public.pop("acquisition_sha256")
    assert declared == hashlib.sha256(
        json.dumps(
            public,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(e.EvidenceBenchAcquisitionError, match="output already exists"):
        e.execute_acquisition_once(
            source_path=source,
            source_binding=binding,
            secret=b"m" * 32,
            protocol_bindings={"synthetic_fixture": True},
            paths=paths,
        )
    assert observed["read_count"] == 1


def test_capacity_shortfall_is_terminal_without_private_pack(tmp_path: Path) -> None:
    payload = _payload()
    for index in range(38):
        payload[index]["paper_as_candidate_pool"] = _sentences(index, 47)
    source = _write_source(tmp_path / "capacity.json", payload)
    paths = _paths(tmp_path)
    receipt = e.execute_acquisition_once(
        source_path=source,
        source_binding=e.hash_source_file(source),
        secret=b"c" * 32,
        protocol_bindings={"synthetic_fixture": True},
        paths=paths,
    )
    assert receipt["status"] == "terminal_source_capacity_insufficient"
    assert receipt["aggregate"]["paper_counts"]["eligible"] == 255
    assert receipt["blocks"]["selected_item_count"] == 0
    assert receipt["blocks"]["smaller_blocks_or_resampling_authorized"] is False
    assert not any(path.exists() for pair in paths.private.values() for path in pair)


def test_post_marker_root_failure_is_terminal_and_formal_api_is_not_a_probe(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "bad-root.json", [{"not": "293 papers"}])
    paths = _paths(tmp_path)
    binding = e.hash_source_file(source)
    with pytest.raises(e.EvidenceBenchAcquisitionError, match="exactly 293"):
        e.execute_acquisition_once(
            source_path=source,
            source_binding=binding,
            secret=b"f" * 32,
            protocol_bindings={"synthetic_fixture": True},
            paths=paths,
        )
    assert paths.marker.is_file()
    assert paths.failure.is_file()
    public = json.loads(paths.public_receipt.read_text(encoding="utf-8"))
    assert public["status"] == "terminal_infrastructure_invalid"
    assert public["aggregate"]["same_source_replay_authorized"] is False

    with pytest.raises(e.EvidenceBenchAcquisitionError, match="only through --formal"):
        e.formal_acquire(
            project=tmp_path,
            source_path=tmp_path / "source",
            selection_secret_path=tmp_path / "secret",
            output_path=tmp_path / "receipt",
        )


def test_source_hash_includes_git_blob_and_rejects_symlink(tmp_path: Path) -> None:
    raw = b"synthetic bytes only"
    source = tmp_path / "source.json"
    source.write_bytes(raw)
    binding = e.hash_source_file(source)
    assert binding.sha256 == hashlib.sha256(raw).hexdigest()
    assert binding.git_blob_sha1 == hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()
    assert binding.byte_size == len(raw)

    symlink = tmp_path / "source-link.json"
    os.symlink(source, symlink)
    with pytest.raises(e.EvidenceBenchAcquisitionError, match="non-symlink"):
        e.hash_source_file(symlink)
