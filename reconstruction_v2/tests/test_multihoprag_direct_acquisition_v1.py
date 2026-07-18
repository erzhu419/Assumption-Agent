from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import pytest

from assumption_agent.benchmarks import multihoprag_direct_acquisition_v1 as acq


def _candidate(index: int, family: str, *, normalized: str | None = None) -> acq.PrivateCandidate:
    normalized_query = normalized or f"synthetic normalized {family} {index}"
    normalized_sha = hashlib.sha256(normalized_query.encode()).hexdigest()
    identity = acq.stable_hash(
        ["synthetic-private-candidate", index, family, normalized_sha]
    )
    return acq.PrivateCandidate(
        source_row_ordinal=128 + index,
        query=f"Synthetic exact {family} question {index}?",
        normalized_query=normalized_query,
        normalized_query_sha256=normalized_sha,
        question_type=family,
        answer=f"synthetic answer {index}",
        gold_article_ids=(1 + (index % 300), 301 + (index % 300)),
        gold_url_identity_sha256=acq.stable_hash(["url", index]),
        evidence_object_hashes=(
            acq.stable_hash(["evidence", index, 0]),
            acq.stable_hash(["evidence", index, 1]),
        ),
        evidence_object_sequence_sha256=acq.stable_hash(
            ["evidence-sequence", index]
        ),
        identity_commitment_sha256=identity,
        source_record_commitment_sha256=acq.stable_hash(
            ["complete-source-record", index, family]
        ),
    )


@pytest.fixture(scope="module")
def private_bundle():
    rows = []
    for family_i, family in enumerate(acq.FAMILIES):
        rows.extend(_candidate(family_i * 1_000 + index, family) for index in range(100))
    blocks, selection_stats = acq.select_private_blocks(
        rows, secret=bytes(range(32))
    )
    articles = tuple(
        acq.CorpusArticle(
            article_id=index,
            exact_url=f"synthetic://article/{index}",
            title=f"Synthetic title {index}",
            author=f"Synthetic author {index % 7}",
            source=f"synthetic-source-{index % 49}",
            published_at=f"2025-{(index % 12) + 1:02d}-01",
            category=f"category-{index % 5}",
            body=f"Synthetic body {index}",
        )
        for index in range(acq.CORPUS_RECORD_COUNT)
    )
    corpus, views, labels = acq.materialize_private_payloads(
        articles=articles, blocks=blocks
    )
    return blocks, selection_stats, corpus, views, labels


@pytest.fixture
def posix_tmp_path():
    # The repository's pytest temp root can resolve to a Windows drvfs mount,
    # which deliberately cannot attest POSIX 0600 modes.  Formal artifacts
    # live on the WSL ext4 project filesystem; /tmp matches those semantics.
    path = Path(tempfile.mkdtemp(prefix="multihoprag-acquisition-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _paths(root: Path) -> acq.AcquisitionPaths:
    return acq.AcquisitionPaths(
        marker=root / acq.ACQUISITION_MARKER_RELATIVE,
        failure=root / acq.FAILURE_RELATIVE,
        corpus_view=root / acq.CORPUS_VIEW_RELATIVE,
        block_views={
            block: root / acq.BLOCK_VIEW_RELATIVES[block]
            for block in acq.BLOCK_ORDER
        },
        block_labels={
            block: root / acq.BLOCK_LABEL_RELATIVES[block]
            for block in ("A_form", "A_hold", "M_search")
        },
        public_receipt=root / acq.PUBLIC_RECEIPT_RELATIVE,
    )


def _promotion(
    acquisition_sha256: str, implementation_sha256: str = "d" * 64
) -> dict[str, object]:
    return acq._with_self_hash(
        {
            "schema": acq.PROMOTION_SCHEMA,
            "version": "v1",
            "status": "A_hold_challenger_promoted",
            "acquisition_sha256": acquisition_sha256,
            "implementation_freeze_sha256": implementation_sha256,
            "f_search_policy_freeze_sha256": "0" * 64,
            "a_hold_action_seal_sha256": "1" * 64,
            "a_hold_output_archive_file_sha256": "4" * 64,
            "a_hold_output_archive_semantic_sha256": "5" * 64,
            "e0_action_id": acq.AGENT_ACTION_IDS[0],
            "e0_policy_sha256": "2" * 64,
            "e1_action_id": acq.AGENT_ACTION_IDS[1],
            "e1_policy_sha256": "3" * 64,
            "a_hold_item_count": acq.BLOCK_COUNTS["A_hold"],
            "a_hold_exact_family_counts": {
                family: acq.FAMILY_QUOTAS["A_hold"] for family in acq.FAMILIES
            },
            "family_balanced_delta_total": [1, 1],
            "one_sided_magnitude_signflip_p": [1, 16],
            "promotion_rule_id": (
                "positive_total_and_one_sided_magnitude_signflip_p_le_0.10"
            ),
            "challenger_promoted": True,
            "outcome_used_to_change_action_evaluator_or_threshold": False,
            "same_source_replay_authorized": False,
        },
        "promotion_sha256",
    )


def _typed_trace(
    action_id: str,
    offset: int = 0,
    *,
    output_top5=None,
    e0_key=(Fraction(1, 2), 10, -1),
    necessary_count: int = 2,
    minimum_leave_one_out_loss: Fraction = Fraction(1, 3),
    path_connectivity: Fraction = Fraction(1, 1),
    graph_sha256: str = "1" * 64,
    plan_sha256: str = "2" * 64,
    query_sha256: str = "3" * 64,
    relevance_sha256: str = "4" * 64,
):
    from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
        ActionTrace,
        CausalSignature,
        CoverageSignature,
        recompute_action_trace_sha256,
    )

    output = tuple(output_top5 or range(offset, offset + 5))
    core = output[:4]
    necessary_fraction = Fraction(necessary_count, 4)
    trace = ActionTrace(
        action_id=action_id,
        output_top5=output,
        core=core,
        core_quality=(1, Fraction(1, 2)),
        coverage=CoverageSignature(
            covered=2,
            total=3,
            value=Fraction(2, 3),
            slot_keys=("a", "b", "c"),
            covered_slot_keys=("a", "b"),
        ),
        causal=CausalSignature(
            necessary_count=necessary_count,
            necessary_fraction=necessary_fraction,
            minimum_leave_one_out_loss=minimum_leave_one_out_loss,
            minimum_replacement_loss=Fraction(0, 1),
            path_connectivity=path_connectivity,
        ),
        e0_key=e0_key,
        e1_key=(
            necessary_fraction,
            minimum_leave_one_out_loss,
            path_connectivity,
            *e0_key,
        ),
        ordered_pair_scan_count=609 * 608,
        extension_scan_count=607 + 606,
        graph_sha256=graph_sha256,
        plan_sha256=plan_sha256,
        query_sha256=query_sha256,
        relevance_sha256=relevance_sha256,
        trace_sha256="0" * 64,
    )
    return replace(trace, trace_sha256=recompute_action_trace_sha256(trace))


def _synthetic_policy_archive(block: str = "F_search"):
    records = []
    trace_matrix = []
    for item_i in range(acq.BLOCK_COUNTS[block]):
        common = {
            "graph_sha256": acq.stable_hash(["graph"]),
            "plan_sha256": acq.stable_hash(["plan", item_i]),
            "query_sha256": acq.stable_hash(["query", item_i]),
            "relevance_sha256": acq.stable_hash(["relevance", item_i]),
        }
        traces = []
        for action_i, action_id in enumerate(acq.AGENT_ACTION_IDS):
            if action_i == 0:
                trace = _typed_trace(
                    action_id,
                    offset=action_i * 5,
                    e0_key=(10, 10, 0),
                    necessary_count=0,
                    minimum_leave_one_out_loss=Fraction(0),
                    path_connectivity=Fraction(0),
                    **common,
                )
            elif action_i == 1:
                trace = _typed_trace(
                    action_id,
                    offset=action_i * 5,
                    e0_key=(1, 1, 0),
                    necessary_count=4,
                    minimum_leave_one_out_loss=Fraction(1),
                    path_connectivity=Fraction(1),
                    **common,
                )
            else:
                trace = _typed_trace(
                    action_id,
                    offset=action_i * 5,
                    e0_key=(0, 0, -action_i),
                    necessary_count=0,
                    minimum_leave_one_out_loss=Fraction(0),
                    path_connectivity=Fraction(0),
                    **common,
                )
            traces.append(acq.encode_typed_action_trace(trace))
        records.append({"agent_action_traces": traces})
        trace_matrix.append([trace["trace_sha256"] for trace in traces])
    return {
        "records": records,
        "agent_complete_six_action_trace_matrix_sha256": acq.stable_hash(
            trace_matrix
        ),
    }


def test_query_normalization_and_length_prefixed_hmac_are_exact():
    assert acq.normalize_query("  Ａ\tStraße\nX  ") == "a strasse x"
    secret = b"s" * 32
    assert acq.hmac_digest(secret, "rank", "ab", "c") != acq.hmac_digest(
        secret, "rank", "a", "bc"
    )
    assert acq.hmac_digest(secret, "rank", "a", "b") != acq.hmac_digest(
        secret, "other-rank", "a", "b"
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError):
        acq.hmac_digest(b"short", "rank", "a")


def test_source_parser_applies_fixed_exclusions_and_private_identity_binding():
    corpus = acq._synthetic_corpus()
    queries = [
        {
            "query": f"Known window {index}?",
            "evidence_list": acq._synthetic_evidence(1, 2),
            "question_type": "comparison_query",
            "answer": "window",
        }
        for index in range(128)
    ]
    queries.extend(
        [
            {
                "query": "References article zero?",
                "evidence_list": acq._synthetic_evidence(0, 2),
                "question_type": "comparison_query",
                "answer": "excluded",
            },
            {
                "query": "Null item?",
                "evidence_list": acq._synthetic_evidence(1, 2),
                "question_type": "null_query",
                "answer": "excluded",
            },
            {
                "query": "Unknown URL?",
                "evidence_list": [
                    {"url": "synthetic://article/1", "fact": "known"},
                    {"url": "synthetic://unknown", "fact": "unknown"},
                ],
                "question_type": "inference_query",
                "answer": "excluded",
            },
            {
                "query": "One deduplicated article?",
                "evidence_list": [
                    {"url": "synthetic://article/1", "fact": "first"},
                    {"url": "synthetic://article/1", "fact": "second"},
                ],
                "question_type": "temporal_query",
                "answer": "excluded",
            },
                {
                    "query": "Valid item?",
                    "evidence_list": [
                        {
                            "url": "synthetic://article/3",
                            "fact": "  ＳＹＮＴＨＥＴＩＣ   FACT FROM ARTICLE 3. ",
                            "rank": 1,
                        },
                        {
                            "url": "synthetic://article/4",
                            "fact": "Synthetic fact from article 4.",
                            "rank": 2,
                        },
                    ],
                    "question_type": "comparison_query",
                    "answer": "valid",
                },
                {
                    "query": "Mismatched evidence fact?",
                    "evidence_list": [
                        {
                            "url": "synthetic://article/5",
                            "fact": "This fact is absent from article five.",
                        },
                        {
                            "url": "synthetic://article/6",
                            "fact": "Synthetic fact from article 6.",
                        },
                    ],
                    "question_type": "inference_query",
                    "answer": "excluded",
                },
        ]
    )
    articles, candidates, stats = acq.parse_source_payloads(
        query_payload=queries, corpus_payload=corpus
    )
    assert len(articles) == 609
    assert len(candidates) == 1
    assert candidates[0].source_row_ordinal == 132
    assert candidates[0].gold_article_ids == (3, 4)
    assert stats["eligibility_exclusion_counts"] == {
        "deduplicated_gold_size_not_2_to_4": 1,
        "evidence_fact_not_in_joined_article_body": 1,
        "exact_url_join_failure": 1,
        "gold_references_exposed_article0": 1,
        "known_public_query_window": 128,
        "null_query": 1,
    }
    assert stats["only_url_and_fact_consumed_for_join_and_gold"] is True
    assert (
        stats["all_eligible_evidence_facts_exact_normalized_body_contained"]
        is True
    )
    first = candidates[0]
    metadata_changed = deepcopy(queries[-2])
    metadata_changed["evidence_list"][0]["rank"] = 99
    _, metadata_candidates, _ = acq.parse_source_payloads(
        query_payload=[*queries[:-2], metadata_changed, queries[-1]],
        corpus_payload=corpus,
    )
    assert metadata_candidates[0].identity_commitment_sha256 == (
        first.identity_commitment_sha256
    )
    assert metadata_candidates[0].source_record_commitment_sha256 != (
        first.source_record_commitment_sha256
    )
    changed = deepcopy(queries[-2])
    changed["question_type"] = "inference_query"
    _, changed_candidates, _ = acq.parse_source_payloads(
        query_payload=[*queries[:-2], changed, queries[-1]],
        corpus_payload=corpus,
    )
    assert changed_candidates[0].identity_commitment_sha256 != first.identity_commitment_sha256


def test_evidence_schema_is_bounded_scalar_only():
    corpus = acq._synthetic_corpus()
    base = [
        {
            "query": f"window {index}",
            "evidence_list": acq._synthetic_evidence(1, 2),
            "question_type": "null_query",
            "answer": "x",
        }
        for index in range(128)
    ]
    nested = {
        "query": "nested metadata",
        "evidence_list": [
            {"url": "synthetic://article/1", "fact": "x", "metadata": {"x": 1}},
            {"url": "synthetic://article/2", "fact": "y"},
        ],
        "question_type": "comparison_query",
        "answer": "x",
    }
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="evidence object schema"):
        acq.parse_source_payloads(
            query_payload=[*base, nested], corpus_payload=corpus
        )


def test_collision_representative_precedes_continuous_family_order():
    secret = bytes(reversed(range(32)))
    rows = []
    for family_i, family in enumerate(acq.FAMILIES):
        rows.extend(_candidate(family_i * 1_000 + index, family) for index in range(100))
    # One cross-family normalized collision proves a group can enter at most
    # one family/block.  Extra capacity keeps every quota feasible.
    rows.append(_candidate(9_999, "comparison_query", normalized=rows[100].normalized_query))
    blocks, stats = acq.select_private_blocks(rows, secret=secret)
    assert stats["selected_block_counts"] == acq.BLOCK_COUNTS
    assert "post_collision_representative_family_counts" not in stats
    selected = [row for block in acq.BLOCK_ORDER for row in blocks[block]]
    assert len({row.normalized_query for row in selected}) == acq.TOTAL_SELECTED
    assert len({row.identity_commitment_sha256 for row in selected}) == acq.TOTAL_SELECTED
    assert sum(
        row.normalized_query == rows[100].normalized_query for row in selected
    ) <= 1

    # For each family, block slices exactly continue the same HMAC order; no
    # block-specific rerank or restarted ranking is permitted.
    for family in acq.FAMILIES:
        reps = {}
        for row in rows:
            current = reps.get(row.normalized_query)
            if current is None or (
                acq.hmac_digest(
                    secret,
                    "collision_representative",
                    row.normalized_query_sha256,
                    row.identity_commitment_sha256,
                ),
                row.identity_commitment_sha256,
                row.source_row_ordinal,
            ) < (
                acq.hmac_digest(
                    secret,
                    "collision_representative",
                    current.normalized_query_sha256,
                    current.identity_commitment_sha256,
                ),
                current.identity_commitment_sha256,
                current.source_row_ordinal,
            ):
                reps[row.normalized_query] = row
        family_order = sorted(
            (row for row in reps.values() if row.question_type == family),
            key=lambda row: (
                acq.hmac_digest(
                    secret,
                    "family_continuous_total_order",
                    family,
                    row.identity_commitment_sha256,
                ),
                row.identity_commitment_sha256,
            ),
        )[:96]
        offset = 0
        for block in acq.BLOCK_ORDER:
            quota = acq.FAMILY_QUOTAS[block]
            observed = {
                row.identity_commitment_sha256
                for row in blocks[block]
                if row.question_type == family
            }
            expected = {
                row.identity_commitment_sha256
                for row in family_order[offset : offset + quota]
            }
            assert observed == expected
            offset += quota


def test_private_views_are_gold_free_F_has_no_labels_and_late_join_is_exact(private_bundle):
    _blocks, _stats, corpus, views, labels = private_bundle
    assert set(corpus["articles"][0]) == {
        "article_id",
        "title",
        "author",
        "source",
        "published_at",
        "category",
        "body",
    }
    assert "synthetic://" not in json.dumps(corpus).casefold()
    assert set(labels) == {"A_form", "A_hold", "M_search"}
    assert "F_search" not in labels
    forbidden = {
        "question_type",
        "answer",
        "evidence_list",
        "url",
        "gold_article_ids",
        "identity_commitment_sha256",
        "source_row_ordinal",
    }
    for block, view in views.items():
        assert not any(key in forbidden for item in view["items"] for key in item)
        if block != "F_search":
            joined = acq.join_late_labels(
                view=view, labels=labels[block], expected_block=block
            )
            assert len(joined) == acq.BLOCK_COUNTS[block]


def test_self_hash_and_late_join_fail_on_tamper(private_bundle):
    _blocks, _stats, _corpus, views, labels = private_bundle
    tampered = deepcopy(views["A_form"])
    tampered["items"][0]["query"] += " tampered"
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="self-hash"):
        acq.join_late_labels(
            view=tampered, labels=labels["A_form"], expected_block="A_form"
        )
    body = dict(tampered)
    del body["block_view_sha256"]
    repaired_hash_only = acq._with_self_hash(body, "block_view_sha256")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="join is incomplete"):
        acq.join_late_labels(
            view=repaired_hash_only,
            labels=labels["A_form"],
            expected_block="A_form",
        )


def test_full_typed_action_trace_codec_recomputes_core_hash_and_rejects_top5_tamper():
    trace = _typed_trace(acq.AGENT_ACTION_IDS[0])
    encoded = acq.encode_typed_action_trace(trace)
    checked, observed = acq._decode_and_verify_action_trace(
        encoded, expected_action_id=acq.AGENT_ACTION_IDS[0]
    )
    assert checked["trace"]["e1"] == [
        [1, 2],
        [1, 3],
        [1, 1],
        [1, 2],
        10,
        -1,
    ]
    assert checked["trace"]["coverage"] == [2, 3]
    assert checked["trace"]["coverage_slot_keys"] == ["a", "b", "c"]
    assert checked["trace"]["coverage_covered_slot_keys"] == ["a", "b"]
    assert observed == trace.trace_sha256
    tampered = deepcopy(encoded)
    tampered["trace"]["output_top5"][-1] = 99
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="SHA256 drifted"):
        acq._decode_and_verify_action_trace(
            tampered, expected_action_id=acq.AGENT_ACTION_IDS[0]
        )
    coverage_tamper = deepcopy(encoded)
    coverage_tamper["trace"]["coverage_slot_keys"][2] = "tampered"
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="SHA256 drifted"):
        acq._decode_and_verify_action_trace(
            coverage_tamper, expected_action_id=acq.AGENT_ACTION_IDS[0]
        )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="record schema"):
        acq._validate_stage_record(
            {},
            block="F_search",
            ordinal=0,
            expected_view_sha256="5" * 64,
        )


def test_stage_record_binds_dense_relevance_raw_and_one_query_observation():
    runtime = {field: "a" * 64 for field in acq.STAGE_RUNTIME_BINDING_KEYS}
    assert acq._validate_stage_runtime_binding(runtime) == runtime
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="schema drifted"):
        acq._validate_stage_runtime_binding(
            {key: value for key, value in runtime.items() if key != "preparation_sha256"}
        )
    query = "  Ｓynthetic\tQuery? "
    query_sha256 = acq._typed_query_sha256(query)
    relevance = tuple([5, 4, 3, 2, 1] + [0] * (acq.CORPUS_RECORD_COUNT - 5))
    _rows, relevance_sha256 = acq._validate_dense_relevance_ints(relevance)
    traces = tuple(
        _typed_trace(
            action_id,
            offset=10 + action_i * 5,
            query_sha256=query_sha256,
            relevance_sha256=relevance_sha256,
        )
        for action_i, action_id in enumerate(acq.AGENT_ACTION_IDS)
    )
    record = acq.build_stage_output_record(
        block="F_search",
        ordinal=0,
        view_sha256="5" * 64,
        dense_relevance_ints=relevance,
        raw_top5=(0, 1, 2, 3, 4),
        hipporag_top5=(5, 6, 7, 8, 9),
        action_traces=traces,
    )
    acq._validate_stage_record(
        record,
        block="F_search",
        ordinal=0,
        expected_view_sha256="5" * 64,
        expected_query_sha256=query_sha256,
        expected_graph_sha256="1" * 64,
    )

    mixed = list(traces)
    mixed[-1] = _typed_trace(
        acq.AGENT_ACTION_IDS[-1],
        offset=40,
        query_sha256="9" * 64,
        relevance_sha256=relevance_sha256,
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="one stage observation"):
        acq.build_stage_output_record(
            block="F_search",
            ordinal=0,
            view_sha256="5" * 64,
            dense_relevance_ints=relevance,
            raw_top5=(0, 1, 2, 3, 4),
            hipporag_top5=(5, 6, 7, 8, 9),
            action_traces=mixed,
        )

    tampered = deepcopy(record)
    tampered["dense_relevance_ints"][10] = 10
    _rows, tampered_sha = acq._validate_dense_relevance_ints(
        tampered["dense_relevance_ints"]
    )
    tampered["relevance_sha256"] = tampered_sha
    del tampered["record_sha256"]
    tampered = acq._with_self_hash(tampered, "record_sha256")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="RAW output differs"):
        acq._validate_stage_record(
            tampered,
            block="F_search",
            ordinal=0,
            expected_view_sha256="5" * 64,
        )


def test_f_policy_freeze_recomputes_actions_and_selection_hashes_authoritatively(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    archive = _synthetic_policy_archive()
    e0, e1, identifiable = acq._recompute_f_search_policy_selections(archive)
    assert identifiable is True
    assert e0.action_id == acq.AGENT_ACTION_IDS[0]
    assert e1.action_id == acq.AGENT_ACTION_IDS[1]

    _blocks, _stats, corpus, views, labels = private_bundle
    commitments = acq.persist_private_payloads(
        corpus=corpus,
        views=views,
        labels=labels,
        paths=_paths(tmp_path),
    )
    receipt = {
        "acquisition_sha256": "a" * 64,
        "private_pack_commitments": commitments,
    }
    archive_binding = {
        "file_sha256": "b" * 64,
        "semantic_sha256": "c" * 64,
        "byte_size": 1,
        "mode": "0600",
    }
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (receipt, {}),
    )
    monkeypatch.setattr(
        acq,
        "load_stage_output_archive",
        lambda **kwargs: (archive, archive_binding),
    )
    freeze = acq.create_f_search_policy_freeze_once(project=tmp_path)
    assert freeze["e0_action_id"] == e0.action_id
    assert freeze["e0_policy_sha256"] == e0.selection_sha256
    assert freeze["e1_action_id"] == e1.action_id
    assert freeze["e1_policy_sha256"] == e1.selection_sha256

    path = tmp_path / acq.F_POLICY_FREEZE_RELATIVE
    tampered = deepcopy(freeze)
    tampered["e0_action_id"] = acq.AGENT_ACTION_IDS[2]
    del tampered["policy_freeze_sha256"]
    tampered = acq._with_self_hash(tampered, "policy_freeze_sha256")
    path.write_bytes(acq._canonical_bytes(tampered) + b"\n")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="binding drifted"):
        acq.load_f_search_policy_freeze(
            project=tmp_path, acquisition=receipt
        )


def test_a_form_policy_freeze_is_prelabel_authoritative_and_F_independent(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    archive = _synthetic_policy_archive("A_form")
    e0, e1, identifiable = acq._recompute_a_form_policy_selections(archive)
    assert identifiable is True
    _blocks, _stats, corpus, views, labels = private_bundle
    commitments = acq.persist_private_payloads(
        corpus=corpus,
        views=views,
        labels=labels,
        paths=_paths(tmp_path),
    )
    receipt = {
        "acquisition_sha256": "a" * 64,
        "private_pack_commitments": commitments,
    }
    archive_binding = {
        "file_sha256": "b" * 64,
        "semantic_sha256": "c" * 64,
        "byte_size": 1,
        "mode": "0600",
    }
    events = []
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (receipt, {}),
    )

    def load_archive(**kwargs):
        assert kwargs["block"] == "A_form"
        events.append("archive")
        return archive, archive_binding

    def load_seal(**kwargs):
        assert kwargs["block"] == "A_form"
        events.append("seal")
        return {"action_seal_sha256": "d" * 64}

    monkeypatch.setattr(acq, "load_stage_output_archive", load_archive)
    monkeypatch.setattr(acq, "load_action_seal", load_seal)
    monkeypatch.setattr(
        acq,
        "load_f_search_policy_freeze",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("A_form must not consume the F capability")
        ),
    )
    freeze = acq.create_a_form_policy_freeze_once(project=tmp_path)
    assert events[:2] == ["archive", "seal"]
    assert freeze["e0_action_id"] == e0.action_id
    assert freeze["e0_policy_sha256"] == e0.selection_sha256
    assert freeze["e1_action_id"] == e1.action_id
    assert freeze["e1_policy_sha256"] == e1.selection_sha256
    assert freeze["selection_purpose"] == "prelabel_descriptive_only_not_F_policy"
    assert freeze["A_form_gold_opened_before_policy_freeze"] is False

    path = tmp_path / acq.A_FORM_POLICY_FREEZE_RELATIVE
    tampered = deepcopy(freeze)
    tampered["e1_action_id"] = acq.AGENT_ACTION_IDS[2]
    del tampered["a_form_policy_freeze_sha256"]
    tampered = acq._with_self_hash(
        tampered, "a_form_policy_freeze_sha256"
    )
    path.write_bytes(acq._canonical_bytes(tampered) + b"\n")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="binding drifted"):
        acq.load_a_form_policy_freeze(
            project=tmp_path, acquisition=receipt
        )


def test_a_form_labels_require_policy_freeze_before_label_path_access(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    _blocks, _stats, corpus, views, labels = private_bundle
    commitments = acq.persist_private_payloads(
        corpus=corpus,
        views=views,
        labels=labels,
        paths=_paths(tmp_path),
    )
    receipt = {
        "acquisition_sha256": "a" * 64,
        "private_pack_commitments": commitments,
    }
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (receipt, {}),
    )
    private_reads = []
    original_read = acq._read_bound_private_json
    monkeypatch.setattr(
        acq,
        "_read_bound_private_json",
        lambda **kwargs: private_reads.append("labels")
        or original_read(**kwargs),
    )
    monkeypatch.setattr(
        acq,
        "load_a_form_policy_freeze",
        lambda **kwargs: (_ for _ in ()).throw(
            acq.MultiHopRAGAcquisitionError("A_form freeze is missing")
        ),
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="freeze is missing"):
        acq.load_block_labels(project=tmp_path, expected_block="A_form")
    assert private_reads == []

    events = []
    monkeypatch.setattr(
        acq,
        "load_a_form_policy_freeze",
        lambda **kwargs: events.append("freeze") or {},
    )
    monkeypatch.setattr(
        acq,
        "_read_bound_private_json",
        lambda **kwargs: events.append("labels") or original_read(**kwargs),
    )
    loaded = acq.load_block_labels(
        project=tmp_path, expected_block="A_form"
    )
    assert loaded == labels["A_form"]
    assert events == ["freeze", "labels"]


def test_persistence_is_0600_and_M_gate_runs_before_path_access(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    _blocks, _stats, corpus, views, labels = private_bundle
    paths = _paths(tmp_path)
    commitments = acq.persist_private_payloads(
        corpus=corpus, views=views, labels=labels, paths=paths
    )
    private_files = [
        paths.corpus_view,
        *paths.block_views.values(),
        *paths.block_labels.values(),
    ]
    assert len(private_files) == commitments["private_file_count"] == 8
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in private_files)
    assert "F_search" not in paths.block_labels

    acquisition_hash = "a" * 64
    receipt = {
        "acquisition_sha256": acquisition_hash,
        "private_pack_commitments": commitments,
    }
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (receipt, {}),
    )

    def sealed_promotion(*, project, acquisition_sha256):
        raise acq.MultiHopRAGAcquisitionError(
            "M_search is sealed before a committed A_hold promotion"
        )

    monkeypatch.setattr(
        acq, "load_committed_promotion_authorization", sealed_promotion
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="sealed before"):
        acq.load_block_view(
            project=tmp_path,
            expected_block="M_search",
        )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="does not exist"):
        acq.load_block_labels(
            project=tmp_path,
            expected_block="F_search",
        )

    authorization_calls = []

    def committed_authorization(*, project, acquisition_sha256):
        authorization_calls.append((project, acquisition_sha256))
        return {"promotion_sha256": "e" * 64}

    monkeypatch.setattr(
        acq,
        "load_committed_promotion_authorization",
        committed_authorization,
    )
    action_seal_calls = []

    def committed_action_seal(*, project, block, acquisition):
        action_seal_calls.append((project, block, acquisition["acquisition_sha256"]))
        return {"action_seal_sha256": "f" * 64}

    monkeypatch.setattr(acq, "load_action_seal", committed_action_seal)
    loaded_view = acq.load_block_view(
        project=tmp_path,
        expected_block="M_search",
    )
    loaded_labels = acq.load_block_labels(
        project=tmp_path,
        expected_block="M_search",
    )
    assert len(
        acq.join_late_labels(
            view=loaded_view,
            labels=loaded_labels,
            expected_block="M_search",
        )
    ) == acq.BLOCK_COUNTS["M_search"]
    assert authorization_calls == [(tmp_path, acquisition_hash)]
    assert action_seal_calls == [(tmp_path, "M_search", acquisition_hash)]


def test_private_file_commitment_rejects_tamper(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    _blocks, _stats, corpus, views, labels = private_bundle
    paths = _paths(tmp_path)
    commitments = acq.persist_private_payloads(
        corpus=corpus, views=views, labels=labels, paths=paths
    )
    receipt = {
        "acquisition_sha256": "c" * 64,
        "private_pack_commitments": commitments,
    }
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (receipt, {}),
    )
    view_path = paths.block_views["A_form"]
    with view_path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="file hash drifted"):
        acq.load_block_view(
            project=tmp_path,
            expected_block="A_form",
        )


@pytest.mark.parametrize("block", ["A_hold", "M_search"])
def test_stage_archive_gate_precedes_every_block_private_path(
    posix_tmp_path, monkeypatch, block
):
    tmp_path = posix_tmp_path
    receipt = {"acquisition_sha256": "a" * 64}
    private_reads = []
    monkeypatch.setattr(
        acq,
        "_read_bound_private_json",
        lambda **kwargs: private_reads.append(kwargs) or {},
    )
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (receipt, {}),
    )
    if block == "A_hold":
        monkeypatch.setattr(
            acq,
            "load_f_search_policy_freeze",
            lambda **kwargs: (_ for _ in ()).throw(
                acq.MultiHopRAGAcquisitionError("A_hold capability sealed")
            ),
        )
    else:
        monkeypatch.setattr(
            acq,
            "load_committed_promotion_authorization",
            lambda **kwargs: (_ for _ in ()).throw(
                acq.MultiHopRAGAcquisitionError("M_search capability sealed")
            ),
        )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="capability sealed"):
        acq.load_stage_output_archive(
            project=tmp_path, block=block, acquisition=receipt
        )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="capability sealed"):
        acq.create_stage_output_archive_once(
            project=tmp_path,
            block=block,
            records=(),
            stage_runtime_binding={field: "b" * 64 for field in acq.STAGE_RUNTIME_BINDING_KEYS},
        )
    assert private_reads == []


def test_secret_marker_precedes_exactly_one_urandom_and_replay_is_burned(posix_tmp_path):
    tmp_path = posix_tmp_path
    calls = []

    def fixed_urandom(count: int) -> bytes:
        marker = tmp_path / "secret.marker"
        assert marker.exists()
        calls.append(count)
        return bytes(range(32))

    implementation = {"implementation_freeze_sha256": "1" * 64}
    query_binding = acq.SourceFileBinding("queries", "2" * 64, "3" * 40, 10)
    corpus_binding = acq.SourceFileBinding("corpus", "4" * 64, "5" * 40, 20)
    previous_umask = os.umask(0o077)
    try:
        payload = acq.create_secret_custody_once(
            marker_path=tmp_path / "secret.marker",
            secret_path=tmp_path / "secret.key",
            public_custody_path=tmp_path / "custody.json",
            implementation_binding=implementation,
            query_binding=query_binding,
            corpus_binding=corpus_binding,
            urandom=fixed_urandom,
        )
    finally:
        os.umask(previous_umask)
    assert calls == [32]
    assert stat.S_IMODE((tmp_path / "secret.marker").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "secret.key").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "custody.json").stat().st_mode) == 0o644
    assert payload["selection_secret_commitment_sha256"] == hashlib.sha256(
        bytes(range(32))
    ).hexdigest()
    acq.verify_self_hash(
        payload,
        hash_field="source_custody_sha256",
        schema=acq.SOURCE_CUSTODY_SCHEMA,
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="replay is forbidden"):
        acq.create_secret_custody_once(
            marker_path=tmp_path / "secret.marker",
            secret_path=tmp_path / "secret.key",
            public_custody_path=tmp_path / "custody.json",
            implementation_binding=implementation,
            query_binding=query_binding,
            corpus_binding=corpus_binding,
            urandom=fixed_urandom,
        )
    assert calls == [32]


def test_exclusive_writer_rejects_indirect_symlink_parent(posix_tmp_path):
    tmp_path = posix_tmp_path
    real = tmp_path / "real"
    real.mkdir()
    indirect = tmp_path / "indirect"
    indirect.symlink_to(real, target_is_directory=True)
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="ancestor is unsafe"):
        acq._write_exclusive(indirect / "child" / "receipt.json", b"{}\n", mode=0o600)
    assert not (real / "child" / "receipt.json").exists()


def test_internal_acquisition_without_capability_has_zero_source_access(
    posix_tmp_path, monkeypatch
):
    tmp_path = posix_tmp_path
    source_reads = []
    monkeypatch.setattr(
        acq,
        "_read_bound_source",
        lambda *args, **kwargs: source_reads.append((args, kwargs)) or b"[]",
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="capability is absent"):
        acq._execute_acquisition_once(
            project=tmp_path,
            capability=object(),
            query_path=tmp_path / acq.QUERY_SOURCE_RELATIVE,
            corpus_path=tmp_path / acq.CORPUS_SOURCE_RELATIVE,
            query_binding=acq.SourceFileBinding("q", "1" * 64, "2" * 40, 2),
            corpus_binding=acq.SourceFileBinding("c", "3" * 64, "4" * 40, 2),
            secret=b"s" * 32,
            custody_binding={"source_custody_sha256": "5" * 64},
            paths=_paths(tmp_path),
        )
    assert source_reads == []
    assert not (tmp_path / acq.ACQUISITION_MARKER_RELATIVE).exists()
    monkeypatch.setattr(acq, "_FORMAL_ENTRY_ACTIVE", True)
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="paths are not canonical"):
        acq._execute_acquisition_once(
            project=tmp_path,
            capability=acq._FORMAL_EXECUTION_CAPABILITY,
            query_path=tmp_path / "wrong-query.json",
            corpus_path=tmp_path / acq.CORPUS_SOURCE_RELATIVE,
            query_binding=acq.SourceFileBinding("q", "1" * 64, "2" * 40, 2),
            corpus_binding=acq.SourceFileBinding("c", "3" * 64, "4" * 40, 2),
            secret=b"s" * 32,
            custody_binding={"source_custody_sha256": "5" * 64},
            paths=_paths(tmp_path),
        )
    assert source_reads == []


def test_failed_randomness_burns_marker_without_secret_rotation(posix_tmp_path):
    tmp_path = posix_tmp_path
    calls = []

    def bad_urandom(count: int) -> bytes:
        calls.append(count)
        return b"bad"

    kwargs = {
        "marker_path": tmp_path / "marker",
        "secret_path": tmp_path / "secret",
        "public_custody_path": tmp_path / "custody",
        "implementation_binding": {"implementation_freeze_sha256": "1" * 64},
        "query_binding": acq.SourceFileBinding("q", "2" * 64, "3" * 40, 1),
        "corpus_binding": acq.SourceFileBinding("c", "4" * 64, "5" * 40, 1),
        "urandom": bad_urandom,
    }
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="did not return"):
        acq.create_secret_custody_once(**kwargs)
    assert (tmp_path / "marker").exists()
    assert not (tmp_path / "secret").exists()
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="replay is forbidden"):
        acq.create_secret_custody_once(**kwargs)
    assert calls == [32]


def test_post_marker_source_failure_is_terminal_and_not_replayable(
    posix_tmp_path, monkeypatch
):
    tmp_path = posix_tmp_path
    query_path = tmp_path / acq.QUERY_SOURCE_RELATIVE
    corpus_path = tmp_path / acq.CORPUS_SOURCE_RELATIVE
    query_path.parent.mkdir(parents=True, exist_ok=True)
    query_path.write_text("[]", encoding="utf-8")
    corpus_path.write_text("[]", encoding="utf-8")
    query_binding = acq.hash_source_file(query_path, logical_name="queries")
    corpus_binding = acq.hash_source_file(corpus_path, logical_name="corpus")
    paths = _paths(tmp_path)
    kwargs = {
        "project": tmp_path,
        "capability": acq._FORMAL_EXECUTION_CAPABILITY,
        "query_path": query_path,
        "corpus_path": corpus_path,
        "query_binding": query_binding,
        "corpus_binding": corpus_binding,
        "secret": b"x" * 32,
        "custody_binding": {"source_custody_sha256": "9" * 64},
        "paths": paths,
    }
    monkeypatch.setattr(acq, "_FORMAL_ENTRY_ACTIVE", True)
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="record count"):
        acq._execute_acquisition_once(**kwargs)
    assert paths.marker.exists()
    assert paths.failure.exists()
    failure = json.loads(paths.failure.read_text(encoding="utf-8"))
    acq.verify_self_hash(
        failure, hash_field="failure_sha256", schema=acq.FAILURE_SCHEMA
    )
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="replay is forbidden"):
        acq._execute_acquisition_once(**kwargs)


def test_implementation_freeze_requires_exact_roles_hashes_and_paths(tmp_path, monkeypatch):
    assert set(acq.FIXED_FREEZE_ROLE_PATHS) == acq.REQUIRED_FREEZE_ROLES
    for role in acq.REQUIRED_FREEZE_ROLES:
        relative = acq.FIXED_FREEZE_ROLE_PATHS.get(role, f"protocol/{role}.txt")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {role}\n", encoding="utf-8")
    bindings = {}
    for role in acq.REQUIRED_FREEZE_ROLES:
        relative = acq.FIXED_FREEZE_ROLE_PATHS.get(role, f"protocol/{role}.txt")
        raw = (tmp_path / relative).read_bytes()
        bindings[role] = {
            "relative_path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": acq._git_blob_sha1(raw),
        }
    freeze = acq._with_self_hash(
        {
            "schema": acq.IMPLEMENTATION_FREEZE_SCHEMA,
            "version": "v1",
            "bindings": bindings,
        },
        "implementation_freeze_sha256",
    )
    freeze_path = tmp_path / acq.IMPLEMENTATION_FREEZE_RELATIVE
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_bytes(acq._canonical_bytes(freeze) + b"\n")

    def fake_head(*, project, relative_paths):
        return "a" * 40, {
            relative: acq._git_blob_sha1((project / relative).read_bytes())
            for relative in relative_paths
        }

    monkeypatch.setattr(acq, "_verify_head_blobs", fake_head)
    observed = acq.verify_committed_implementation_freeze(tmp_path)
    assert observed["required_role_count"] == len(acq.REQUIRED_FREEZE_ROLES)
    assert observed["all_bindings_byte_match_committed_HEAD"] is True

    for role in sorted(acq.REQUIRED_FREEZE_ROLES):
        wrong = deepcopy(freeze)
        wrong_relative = f"wrong/{role}.txt"
        wrong_path = tmp_path / wrong_relative
        wrong_path.parent.mkdir(parents=True, exist_ok=True)
        original = (tmp_path / acq.FIXED_FREEZE_ROLE_PATHS[role]).read_bytes()
        wrong_path.write_bytes(original)
        wrong["bindings"][role]["relative_path"] = wrong_relative
        del wrong["implementation_freeze_sha256"]
        wrong = acq._with_self_hash(wrong, "implementation_freeze_sha256")
        freeze_path.write_bytes(acq._canonical_bytes(wrong) + b"\n")
        with pytest.raises(acq.MultiHopRAGAcquisitionError, match="frozen path"):
            acq.verify_committed_implementation_freeze(tmp_path)
    freeze_path.write_bytes(acq._canonical_bytes(freeze) + b"\n")

    bad = deepcopy(freeze)
    del bad["bindings"]["formal_runner"]
    del bad["implementation_freeze_sha256"]
    bad = acq._with_self_hash(bad, "implementation_freeze_sha256")
    freeze_path.write_bytes(acq._canonical_bytes(bad) + b"\n")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="role set"):
        acq.verify_committed_implementation_freeze(tmp_path)


def test_formal_entrypoints_reject_cross_checkout_before_any_marker(
    posix_tmp_path, monkeypatch
):
    tmp_path = posix_tmp_path
    expected_module = tmp_path / acq.ACQUISITION_RELATIVE
    expected_module.parent.mkdir(parents=True, exist_ok=True)
    expected_module.write_text("# checkout B placeholder\n", encoding="ascii")
    implementation = {
        "implementation_freeze_sha256": "1" * 64,
        "all_bindings_byte_match_committed_HEAD": True,
        "required_role_count": len(acq.REQUIRED_FREEZE_ROLES),
    }
    monkeypatch.setattr(acq, "_require_private_artifacts_ignored", lambda root: None)
    monkeypatch.setattr(
        acq,
        "verify_committed_implementation_freeze",
        lambda project: implementation,
    )

    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="outside the frozen"):
        acq.create_source_custody(tmp_path)
    assert not (tmp_path / acq.SECRET_MARKER_RELATIVE).exists()
    assert not (tmp_path / acq.SELECTION_SECRET_RELATIVE).exists()

    monkeypatch.setattr(acq, "_FORMAL_ENTRY_ACTIVE", True)
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="outside the frozen"):
        acq.formal_acquire(tmp_path)
    assert not (tmp_path / acq.ACQUISITION_MARKER_RELATIVE).exists()


def test_git_blob_verification_uses_immutable_oid_and_detects_head_drift(
    posix_tmp_path, monkeypatch
):
    tmp_path = posix_tmp_path
    (tmp_path / ".git").mkdir()
    bound = tmp_path / "bound.txt"
    bound.write_text("bound\n", encoding="utf-8")
    oid = acq._git_blob_sha1(bound.read_bytes())
    first_head = "a" * 40
    second_head = "b" * 40
    calls = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = b""
            self.returncode = 0

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        if args[3] == "rev-parse" and len(calls) == 1:
            return Result((first_head + "\n").encode())
        if args[3] == "ls-tree":
            assert first_head in args
            assert "HEAD" not in args
            return Result(f"100644 blob {oid}\tbound.txt\0".encode())
        return Result((second_head + "\n").encode())

    monkeypatch.setattr(acq.subprocess, "run", fake_run)
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="HEAD drifted"):
        acq._verify_head_blobs(
            project=tmp_path, relative_paths=("bound.txt",)
        )
    assert len(calls) == 3


def test_acquisition_receipt_reverifies_committed_source_custody_chain(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    _blocks, _stats, corpus, views, labels = private_bundle
    commitments = acq.persist_private_payloads(
        corpus=corpus,
        views=views,
        labels=labels,
        paths=_paths(tmp_path),
    )
    source_bindings = {
        "query_source": {
            "logical_name": acq.QUERY_SOURCE_NAME,
            "sha256": "1" * 64,
            "git_blob_sha1": acq.QUERY_SOURCE_GIT_BLOB_SHA1,
            "byte_size": acq.QUERY_SOURCE_SIZE,
        },
        "corpus_source": {
            "logical_name": acq.CORPUS_SOURCE_NAME,
            "sha256": "2" * 64,
            "git_blob_sha1": acq.CORPUS_SOURCE_GIT_BLOB_SHA1,
            "byte_size": acq.CORPUS_SOURCE_SIZE,
        },
    }
    dataset_identity = {
        "repository": acq.DATASET_REPOSITORY,
        "dataset_commit": acq.DATASET_COMMIT,
        "code_commit": acq.CODE_COMMIT,
        "license": "ODC-BY",
    }
    custody = {
        "dataset_identity": dataset_identity,
        "source_bindings": source_bindings,
    }
    custody_binding = {
        "source_custody_sha256": "3" * 64,
        "source_custody_file_sha256": "4" * 64,
        "source_custody_git_blob_sha1": "5" * 40,
        "custody_verified_at_git_HEAD": "6" * 40,
    }
    marker = acq._with_self_hash(
        {
            "schema": acq.ATTEMPT_MARKER_SCHEMA,
            "version": acq.VERSION,
            "phase": "formal_source_parse_and_block_formation",
            "bindings": {
                "source_custody_sha256": custody_binding[
                    "source_custody_sha256"
                ],
                "query_source_sha256": source_bindings["query_source"]["sha256"],
                "corpus_source_sha256": source_bindings["corpus_source"]["sha256"],
            },
            "replay_secret_rotation_resample_replacement_or_retry_authorized": False,
        },
        "marker_sha256",
    )
    acq._write_json_exclusive(
        tmp_path / acq.ACQUISITION_MARKER_RELATIVE, marker, mode=0o600
    )

    def make_receipt(*, sources=source_bindings, binding=custody_binding):
        return acq._with_self_hash(
            {
                "schema": acq.PUBLIC_RECEIPT_SCHEMA,
                "version": acq.VERSION,
                "status": "private_four_block_pack_formed",
                "dataset_identity": dataset_identity,
                "custody_binding": binding,
                "attempt_marker_sha256": marker["marker_sha256"],
                "source_bindings": sources,
                "source_qualification": {},
                "selection_qualification": {},
                "private_pack_commitments": commitments,
                "label_isolation": {},
                "public_candidate_identity_query_answer_fact_URL_evidence_or_gold": False,
                "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized": False,
            },
            "acquisition_sha256",
        )

    receipt_path = tmp_path / acq.PUBLIC_RECEIPT_RELATIVE
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    def write_receipt(payload):
        receipt_path.write_bytes(acq._canonical_bytes(payload) + b"\n")
        os.chmod(receipt_path, 0o644)

    def fake_head(*, project, relative_paths):
        return "7" * 40, {
            relative: acq._git_blob_sha1((project / relative).read_bytes())
            for relative in relative_paths
        }

    monkeypatch.setattr(acq, "_verify_head_blobs", fake_head)
    monkeypatch.setattr(
        acq,
        "verify_committed_source_custody",
        lambda project: (custody, custody_binding),
    )
    good = make_receipt()
    write_receipt(good)
    loaded, _binding = acq.load_committed_acquisition_receipt(tmp_path)
    assert loaded["acquisition_sha256"] == good["acquisition_sha256"]

    broken_sources = deepcopy(source_bindings)
    broken_sources["query_source"]["sha256"] = "8" * 64
    write_receipt(make_receipt(sources=broken_sources))
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="custody chain"):
        acq.load_committed_acquisition_receipt(tmp_path)

    fake_binding = dict(custody_binding)
    fake_binding["source_custody_file_sha256"] = "9" * 64
    write_receipt(make_receipt(binding=fake_binding))
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="custody chain"):
        acq.load_committed_acquisition_receipt(tmp_path)


def test_M_authorization_must_be_canonical_committed_and_pass_frozen_rule(
    posix_tmp_path, monkeypatch
):
    tmp_path = posix_tmp_path
    implementation_sha = "d" * 64
    acquisition = {"acquisition_sha256": "b" * 64}
    (tmp_path / "manifests").mkdir(parents=True, exist_ok=True)
    promotion = _promotion(
        acquisition["acquisition_sha256"], implementation_sha
    )
    promotion_path = tmp_path / acq.PROMOTION_RELATIVE
    promotion_path.write_bytes(acq._canonical_bytes(promotion) + b"\n")

    def fake_head(*, project, relative_paths):
        return "a" * 40, {
            relative: acq._git_blob_sha1((project / relative).read_bytes())
            for relative in relative_paths
        }

    monkeypatch.setattr(acq, "_verify_head_blobs", fake_head)
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (acquisition, {}),
    )
    monkeypatch.setattr(
        acq,
        "verify_committed_implementation_freeze",
        lambda project: {"implementation_freeze_sha256": implementation_sha},
    )
    monkeypatch.setattr(
        acq,
        "_assess_a_hold_promotion",
        lambda **kwargs: {
            "challenger_promoted": True,
            "f_search_policy_freeze_sha256": promotion[
                "f_search_policy_freeze_sha256"
            ],
            "a_hold_action_seal_sha256": promotion[
                "a_hold_action_seal_sha256"
            ],
            "a_hold_output_archive_file_sha256": promotion[
                "a_hold_output_archive_file_sha256"
            ],
            "a_hold_output_archive_semantic_sha256": promotion[
                "a_hold_output_archive_semantic_sha256"
            ],
            "e0_action_id": promotion["e0_action_id"],
            "e1_action_id": promotion["e1_action_id"],
            "e0_policy_sha256": promotion["e0_policy_sha256"],
            "e1_policy_sha256": promotion["e1_policy_sha256"],
            "family_balanced_delta_total": Fraction(1, 1),
            "one_sided_magnitude_signflip_p": Fraction(1, 16),
        },
    )
    authorization = acq.load_committed_promotion_authorization(
        project=tmp_path,
        acquisition_sha256=acquisition["acquisition_sha256"],
    )
    assert authorization["promotion_sha256"] == promotion["promotion_sha256"]
    assert authorization["e0_policy_sha256"] != authorization["e1_policy_sha256"]

    failed = deepcopy(promotion)
    failed["one_sided_magnitude_signflip_p"] = [1, 5]
    del failed["promotion_sha256"]
    failed = acq._with_self_hash(failed, "promotion_sha256")
    promotion_path.write_bytes(acq._canonical_bytes(failed) + b"\n")
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="do not pass"):
        acq.load_committed_promotion_authorization(
            project=tmp_path,
            acquisition_sha256=acquisition["acquisition_sha256"],
        )


def test_a_hold_assessment_and_promotion_are_canonical_not_caller_reported(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    _blocks, _stats, _corpus, views, labels = private_bundle
    records = []
    for ordinal, label in enumerate(labels["A_hold"]["items"]):
        gold = tuple(label["gold_article_ids"])
        misses = tuple(
            index
            for index in range(acq.CORPUS_RECORD_COUNT)
            if index not in gold
        )[:5]
        challenger = tuple(
            [*gold]
            + [
                index
                for index in range(acq.CORPUS_RECORD_COUNT)
                if index not in gold
            ][: 5 - len(gold)]
        )
        common = {
            "graph_sha256": acq.stable_hash(["A_hold-graph"]),
            "plan_sha256": acq.stable_hash(["A_hold-plan", ordinal]),
            "query_sha256": acq.stable_hash(["A_hold-query", ordinal]),
            "relevance_sha256": acq.stable_hash(["A_hold-relevance", ordinal]),
        }
        traces = []
        for action_i, action_id in enumerate(acq.AGENT_ACTION_IDS):
            output = challenger if action_i == 1 else misses
            traces.append(
                acq.encode_typed_action_trace(
                    _typed_trace(action_id, output_top5=output, **common)
                )
            )
        records.append({"agent_action_traces": traces})
    acquisition = {"acquisition_sha256": "a" * 64}
    freeze = {
        "e0_action_id": acq.AGENT_ACTION_IDS[0],
        "e1_action_id": acq.AGENT_ACTION_IDS[1],
        "e0_policy_sha256": "2" * 64,
        "e1_policy_sha256": "3" * 64,
        "policy_freeze_sha256": "0" * 64,
    }
    archive_binding = {
        "file_sha256": "4" * 64,
        "semantic_sha256": "5" * 64,
    }
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (acquisition, {}),
    )
    monkeypatch.setattr(acq, "load_f_search_policy_freeze", lambda **kwargs: freeze)
    monkeypatch.setattr(
        acq,
        "load_action_seal",
        lambda **kwargs: {"action_seal_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        acq,
        "load_stage_output_archive",
        lambda **kwargs: ({"records": records}, archive_binding),
    )
    monkeypatch.setattr(
        acq,
        "load_block_view",
        lambda **kwargs: views["A_hold"],
    )
    monkeypatch.setattr(
        acq,
        "load_block_labels",
        lambda **kwargs: labels["A_hold"],
    )
    decision = acq.assess_a_hold_promotion(project=tmp_path)
    assert decision["challenger_promoted"] is True
    assert decision["family_balanced_delta_total"] > 0
    assert decision["one_sided_magnitude_signflip_p"] <= Fraction(1, 10)

    monkeypatch.setattr(
        acq,
        "verify_committed_implementation_freeze",
        lambda project: {"implementation_freeze_sha256": "d" * 64},
    )
    monkeypatch.setattr(acq, "_assess_a_hold_promotion", lambda **kwargs: decision)
    promotion = acq.create_a_hold_promotion_once(project=tmp_path)
    assert promotion["family_balanced_delta_total"] == decision[
        "family_balanced_delta_total"
    ]
    assert promotion["e0_action_id"] == acq.AGENT_ACTION_IDS[0]
    assert promotion["e1_action_id"] == acq.AGENT_ACTION_IDS[1]


def test_m_search_assessment_recomputes_all_exact_boundaries(
    posix_tmp_path, private_bundle, monkeypatch
):
    tmp_path = posix_tmp_path
    _blocks, _stats, _corpus, views, labels = private_bundle

    def method(method_id, output):
        return acq._with_self_hash(
            {"method": method_id, "terminal": True, "output_top5": list(output)},
            "output_sha256",
        )

    records = []
    for ordinal, label in enumerate(labels["M_search"]["items"]):
        gold = tuple(label["gold_article_ids"])
        misses = tuple(
            index
            for index in range(acq.CORPUS_RECORD_COUNT)
            if index not in gold
        )[:5]
        challenger = tuple(
            [*gold]
            + [
                index
                for index in range(acq.CORPUS_RECORD_COUNT)
                if index not in gold
            ][: 5 - len(gold)]
        )
        common = {
            "graph_sha256": acq.stable_hash(["M-graph"]),
            "plan_sha256": acq.stable_hash(["M-plan", ordinal]),
            "query_sha256": acq.stable_hash(["M-query", ordinal]),
            "relevance_sha256": acq.stable_hash(["M-relevance", ordinal]),
        }
        traces = [
            acq.encode_typed_action_trace(
                _typed_trace(
                    action_id,
                    output_top5=challenger if action_i == 1 else misses,
                    **common,
                )
            )
            for action_i, action_id in enumerate(acq.AGENT_ACTION_IDS)
        ]
        records.append(
            {
                "agent_action_traces": traces,
                "raw_output": method("RAW", misses),
                "hipporag_output": method("HippoRAG", misses),
            }
        )
    acquisition = {"acquisition_sha256": "a" * 64}
    promotion = {
        "promotion_sha256": "6" * 64,
        "e0_action_id": acq.AGENT_ACTION_IDS[0],
        "e1_action_id": acq.AGENT_ACTION_IDS[1],
    }
    monkeypatch.setattr(
        acq,
        "load_committed_acquisition_receipt",
        lambda project: (acquisition, {}),
    )
    monkeypatch.setattr(
        acq,
        "load_committed_promotion_authorization",
        lambda **kwargs: promotion,
    )
    monkeypatch.setattr(
        acq,
        "load_action_seal",
        lambda **kwargs: {"action_seal_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        acq,
        "load_stage_output_archive",
        lambda **kwargs: (
            {"records": records},
            {"file_sha256": "8" * 64, "semantic_sha256": "9" * 64},
        ),
    )
    monkeypatch.setattr(
        acq, "load_block_view", lambda **kwargs: views["M_search"]
    )
    monkeypatch.setattr(
        acq, "load_block_labels", lambda **kwargs: labels["M_search"]
    )
    result = acq.assess_m_search(project=tmp_path)
    assert result["l5_passed"] is True
    assert result["cross_family_agent_over_hippo_passed"] is True
    assert result["agent_minus_raw_delta_total"] > 0
    assert result["agent_minus_raw_complete_delta"] > 0
    assert result["raw_complete_advantage_overcome"] is True


def test_public_receipt_leak_guard_and_synthetic_qualification():
    with pytest.raises(acq.MultiHopRAGAcquisitionError, match="private field answer"):
        acq._assert_public_safe({"aggregate": {}, "answer": "leaked"})
    receipt = acq.run_synthetic_qualification()
    assert receipt["status"] == "pass"
    assert receipt["network_or_official_source_access"] is False
    assert receipt["F_search_label_pack_created"] is False
    assert (
        receipt[
            "all_eligible_evidence_facts_exact_normalized_body_contained"
        ]
        is True
    )
    assert receipt["selected_block_counts"] == acq.BLOCK_COUNTS
    acq.verify_self_hash(receipt, hash_field="qualification_sha256")
