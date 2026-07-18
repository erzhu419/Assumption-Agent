from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as module
from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic


SECRET = bytes(range(32))


def _evidence_key(stem: str, ordinal: int) -> str:
    return f"Synthetic_{stem}_sentence_{ordinal}"


def _candidate_pool() -> list[module.CandidateRecord]:
    rows: list[module.CandidateRecord] = []
    for family_i, family in enumerate(module.FAMILIES):
        for verdict_i, verdict in enumerate(module.VERDICTS):
            for ordinal in range(38):
                stem = f"f{family_i}-v{verdict_i}-r{ordinal}"
                first = (_evidence_key(stem, 0), _evidence_key(stem, 1))
                second = (_evidence_key(stem, 1), _evidence_key(stem, 2))
                rows.append(
                    module.CandidateRecord(
                        source_key=stem,
                        claim=f"Synthetic claim {family_i} {verdict_i} {ordinal}",
                        family=family,
                        verdict=verdict,
                        evidence_sets=(first, second),
                        all_official_evidence_keys=(
                            _evidence_key(stem, 0),
                            _evidence_key(stem, 1),
                            _evidence_key(stem, 2),
                            _evidence_key(stem, 3),
                        ),
                    )
                )
    rows.extend(
        [
            module.CandidateRecord(
                source_key="collision-a",
                claim="  Normalized   collision ",
                family=module.FAMILIES[0],
                verdict=module.VERDICTS[0],
                evidence_sets=((
                    _evidence_key("collision-a", 0),
                    _evidence_key("collision-a", 1),
                ),),
                all_official_evidence_keys=(
                    _evidence_key("collision-a", 0),
                    _evidence_key("collision-a", 1),
                ),
            ),
            module.CandidateRecord(
                source_key="collision-b",
                claim="normalized collision",
                family=module.FAMILIES[0],
                verdict=module.VERDICTS[0],
                evidence_sets=((
                    _evidence_key("collision-b", 0),
                    _evidence_key("collision-b", 1),
                ),),
                all_official_evidence_keys=(
                    _evidence_key("collision-b", 0),
                    _evidence_key("collision-b", 1),
                ),
            ),
        ]
    )
    return rows


def _selected() -> tuple[
    dict[str, tuple[module.AssignedRecord, ...]], dict[str, object]
]:
    return module.select_private_blocks(_candidate_pool(), SECRET)


def _unit(key: str) -> module.CorpusUnit:
    parsed = module.parse_element_id(key)
    return module.CorpusUnit(
        unit_key=key,
        text=f"TARGET: {key}\nTYPE: sentence",
        unit_type="sentence",
        sidecar={
            "linearizer_version": atomic.VERSION,
            "page": parsed.page,
            "local_id": parsed.local_id,
            "unit_type": "sentence",
            "official_ordinal": parsed.indices[0],
        },
    )


def _source_order(rows: list[module.CorpusUnit]) -> list[module.CorpusUnit]:
    return sorted(
        rows,
        key=lambda row: (row.page, row.sidecar["official_ordinal"]),
    )


def test_hmac_selection_is_balanced_unique_and_collision_safe() -> None:
    blocks, stats = _selected()
    assert set(blocks) == set(module.BLOCK_ORDER)
    assert stats["normalized_claim_collision_group_count"] == 1
    assert stats["normalized_claim_collision_record_count"] == 2
    selected = [row for block in module.BLOCK_ORDER for row in blocks[block]]
    assert len(selected) == sum(module.BLOCK_COUNTS.values()) == 288
    assert len({row.record.source_key for row in selected}) == 288
    assert len({row.record.normalized_claim for row in selected}) == 288
    assert not any(row.record.source_key.startswith("collision") for row in selected)
    for block, rows in blocks.items():
        assert len(rows) == module.BLOCK_COUNTS[block]
        counts = Counter((row.record.family, row.record.verdict) for row in rows)
        for family in module.FAMILIES:
            for verdict in module.VERDICTS:
                assert counts[(family, verdict)] == module.PER_FAMILY[block] // 2
        assert [row.ordinal for row in rows] == list(range(len(rows)))


def test_selection_is_repeat_exact_and_secret_sensitive() -> None:
    first, _ = _selected()
    second, _ = _selected()
    assert first == second
    changed, _ = module.select_private_blocks(
        _candidate_pool(), bytes(reversed(range(32)))
    )
    assert {
        block: tuple(row.record.source_key for row in first[block])
        for block in module.BLOCK_ORDER
    } != {
        block: tuple(row.record.source_key for row in changed[block])
        for block in module.BLOCK_ORDER
    }


def test_fixed_corpus_includes_gold_and_excludes_known_alternatives() -> None:
    blocks, _ = _selected()
    selected = [row for block in module.BLOCK_ORDER for row in blocks[block]]
    official = {
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    }
    gold = {key for row in selected for key in row.canonical_gold_keys}
    alternatives = official - gold
    rows = [_unit(key) for key in sorted(official)]
    rows.extend(
        _unit(f"Synthetic_Distractor_{i:05d}_sentence_0")
        for i in range(9000)
    )
    corpus, index, stats = module.build_fixed_corpus(
        blocks=blocks,
        units=_source_order(rows),
        secret=SECRET,
    )
    live = {row.unit_key for row in corpus}
    assert len(corpus) == len(index) == module.CORPUS_UNIT_COUNT
    assert gold.issubset(live)
    assert live.isdisjoint(alternatives)
    assert stats["all_selected_canonical_gold_included"] is True
    assert stats["gold_origin_serialized_in_corpus"] is False
    assert stats["source_atomic_unit_scan_count"] == len(rows)
    assert len(stats["source_atomic_unit_stream_sha256"]) == 64


def test_private_packs_keep_F_labels_absent_and_views_gold_free() -> None:
    blocks, _ = _selected()
    selected = [row for block in module.BLOCK_ORDER for row in blocks[block]]
    official = {
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    }
    rows = [_unit(key) for key in sorted(official)]
    rows.extend(
        _unit(f"Synthetic_Distractor_{i:05d}_sentence_0")
        for i in range(9000)
    )
    corpus, index, _ = module.build_fixed_corpus(
        blocks=blocks,
        units=_source_order(rows),
        secret=SECRET,
    )
    corpus_view, views, labels = module.materialize_private_payloads(
        blocks=blocks,
        corpus=corpus,
        corpus_index=index,
    )
    assert module.verify_self_hash(corpus_view, "corpus_view_sha256")
    assert set(views) == set(module.BLOCK_ORDER)
    assert set(labels) == {"A_form", "A_hold", "M_search"}
    assert corpus_view["gold_origin_or_membership_included"] is False
    assert all(
        "gold_unit_indices" not in unit and "is_gold" not in unit
        for unit in corpus_view["units"]
    )
    for block, view in views.items():
        assert module.verify_self_hash(view, "block_view_sha256")
        serialized = json.dumps(view, sort_keys=True)
        assert "family" not in serialized
        assert "verdict" not in serialized
        assert "evidence" not in serialized
        assert "block" not in view
        assert all(set(item) == {"claim"} for item in view["items"])
        assert view["late_label_fields_included"] is False
        if block in labels:
            assert module.verify_self_hash(
                labels[block], "block_labels_sha256"
            )


def test_receipt_tamper_and_malformed_candidate_fail_closed() -> None:
    body = module.self_hashed(
        {"schema": "synthetic", "value": 1}, "receipt_sha256"
    )
    changed = copy.deepcopy(body)
    changed["value"] = 2
    with pytest.raises(module.FeverousP6E2AcquisitionError):
        module.verify_self_hash(changed, "receipt_sha256")
    with pytest.raises(module.FeverousP6E2AcquisitionError):
        module.CandidateRecord(
            source_key="bad",
            claim="Bad",
            family=module.FAMILIES[0],
            verdict=module.VERDICTS[0],
            evidence_sets=(("only-one",),),
            all_official_evidence_keys=("only-one",),
        )


def test_corpus_rejects_wrong_index_forbidden_sidecar_and_mutation() -> None:
    blocks, _ = _selected()
    selected = [row for block in module.BLOCK_ORDER for row in blocks[block]]
    official = {
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    }
    rows = [_unit(key) for key in sorted(official)]
    rows.extend(
        _unit(f"Synthetic_Distractor_{i:05d}_sentence_0")
        for i in range(9000)
    )
    corpus, index, _ = module.build_fixed_corpus(
        blocks=blocks,
        units=_source_order(rows),
        secret=SECRET,
    )
    wrong_index = dict(index)
    first, second = list(wrong_index)[:2]
    wrong_index[first], wrong_index[second] = (
        wrong_index[second],
        wrong_index[first],
    )
    with pytest.raises(module.FeverousP6E2AcquisitionError, match="exact corpus"):
        module.materialize_private_payloads(
            blocks=blocks,
            corpus=corpus,
            corpus_index=wrong_index,
        )

    key = "Synthetic_Leak_sentence_0"
    with pytest.raises(module.FeverousP6E2AcquisitionError, match="source-only"):
        module.CorpusUnit(
            unit_key=key,
            text="TARGET: leak\nTYPE: sentence",
            unit_type="sentence",
            sidecar={
                "linearizer_version": atomic.VERSION,
                "page": "Synthetic_Leak",
                "local_id": "sentence_0",
                "unit_type": "sentence",
                "gold": True,
            },
        )

    mutable_sets = [[
        _evidence_key("immutable", 0),
        _evidence_key("immutable", 1),
    ]]
    mutable_universe = list(mutable_sets[0])
    record = module.CandidateRecord(
        source_key="immutable",
        claim="Immutable claim",
        family=module.FAMILIES[0],
        verdict=module.VERDICTS[0],
        evidence_sets=mutable_sets,  # type: ignore[arg-type]
        all_official_evidence_keys=mutable_universe,  # type: ignore[arg-type]
    )
    mutable_sets[0].append(_evidence_key("immutable", 2))
    mutable_universe.append(_evidence_key("immutable", 2))
    assert len(record.evidence_sets[0]) == 2
    assert len(record.all_official_evidence_keys) == 2


def test_corpus_builder_consumes_a_one_pass_ordered_stream_and_rejects_drift() -> None:
    blocks, _ = _selected()
    selected = [row for block in module.BLOCK_ORDER for row in blocks[block]]
    official = {
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    }
    rows = [_unit(key) for key in sorted(official)]
    rows.extend(
        _unit(f"Synthetic_Distractor_{i:05d}_sentence_0")
        for i in range(9000)
    )
    ordered = _source_order(rows)
    corpus, index, stats = module.build_fixed_corpus(
        blocks=blocks,
        units=(row for row in ordered),
        secret=SECRET,
    )
    assert len(corpus) == len(index) == module.CORPUS_UNIT_COUNT
    assert stats["source_atomic_unit_scan_count"] == len(ordered)
    swapped = list(ordered)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(module.FeverousP6E2AcquisitionError, match="source stream"):
        module.build_fixed_corpus(
            blocks=blocks,
            units=iter(swapped),
            secret=SECRET,
        )
