from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json

import pytest

from assumption_agent.benchmarks import techqa_p1_formal_v1 as formal


def _family_question_text(family: str, token: str) -> tuple[str, str]:
    if family == formal.INFORMATION:
        return (
            f"Reference details for {token}",
            f"Configuration metadata concerning {token}.",
        )
    if family == formal.PROCEDURE:
        return (
            f"How to configure {token}",
            f"Steps for installing {token}.",
        )
    if family == formal.TROUBLESHOOT:
        return (
            f"Fix error {token}",
            f"The component {token} cannot start.",
        )
    raise AssertionError(family)


def _source(*, reverse_candidates: bool = False) -> formal.VerifiedSource:
    shared = [
        formal.VerifiedDocument(
            document_id=f"shared-{index:02d}",
            title=f"Shared technote {index}",
            text=f"General unrelated support material number {index}.",
        )
        for index in range(49)
    ]
    documents = list(shared)
    training: list[formal.VerifiedQuestion] = []
    dev: list[formal.VerifiedQuestion] = []
    for split, per_family, target in (
        ("train", 48, training),
        ("dev", 24, dev),
    ):
        for family in formal.FAMILY_IDS:
            for index in range(per_family):
                token = (
                    f"{split}-{family.casefold()}-unique-{index:03d}"
                )
                gold_id = f"gold-{token}"
                title, text = _family_question_text(family, token)
                documents.append(
                    formal.VerifiedDocument(
                        document_id=gold_id,
                        title=f"Technote for {token}",
                        text=f"Exact answer material for {token}.",
                    )
                )
                candidate_ids = [
                    row.document_id for row in shared
                ] + [gold_id]
                if reverse_candidates:
                    candidate_ids.reverse()
                target.append(
                    formal.VerifiedQuestion(
                        question_id=f"question-{token}",
                        question_title=title,
                        question_text=text,
                        document_ids=tuple(candidate_ids),
                        gold_document_id=gold_id,
                    )
                )
    return formal.VerifiedSource(
        training_questions=tuple(training),
        dev_questions=tuple(dev),
        documents=tuple(documents),
        commitments=formal.SourceCommitments(
            training_q_a_sha256="1" * 64,
            dev_q_a_sha256="2" * 64,
            training_dev_technotes_sha256="3" * 64,
            qualification_receipt_sha256="4" * 64,
        ),
    )


@pytest.fixture(scope="module")
def source() -> formal.VerifiedSource:
    return _source()


@pytest.fixture(scope="module")
def prepared(source: formal.VerifiedSource) -> formal.PreparedStudy:
    return formal.prepare_formal_study(
        source, hmac_secret=b"s" * formal.HMAC_SECRET_BYTES
    )


def _hippo_results(
    prepared: formal.PreparedStudy,
) -> tuple[formal.OfficialHippoResult, ...]:
    actions = prepared.a_hold.action_by_work_id
    values = []
    for request in prepared.hippo_requests:
        ordinals = actions[
            request.work_id
        ].e0.top5_document_ordinals
        document_ids = tuple(
            str(request.documents[value]["document_id"])
            for value in ordinals
        )
        values.append(
            formal.OfficialHippoResult(
                block=request.block,
                cluster_index=request.cluster_index,
                work_id=request.work_id,
                input_sha256=request.input_sha256,
                query_bytes_sha256=request.query_bytes_sha256,
                serialized_document_set_sha256=(
                    request.serialized_document_set_sha256
                ),
                # Exercise the document-ID binding path.
                top5_document_ids=document_ids,
            )
        )
    return tuple(values)


def _comparison_rows(
    *,
    failed_cluster: int | None = None,
) -> list[tuple[str, int, Fraction, Fraction]]:
    rows = []
    for cluster in range(4):
        for family in formal.FAMILY_IDS:
            left = Fraction(1, 1)
            right = Fraction(0, 1)
            if cluster == failed_cluster:
                left = right
            rows.append((family, cluster, left, right))
    return rows


def test_operational_classifier_is_frozen_and_troubleshoot_has_priority() -> None:
    assert formal.operational_family(
        "How to fix a server", "Steps for an error"
    ) == formal.TROUBLESHOOT
    assert formal.operational_family(
        "How can I configure a server?", "Instructions follow."
    ) == formal.PROCEDURE
    assert formal.operational_family(
        "Server edition matrix", "Compatibility metadata."
    ) == formal.INFORMATION
    # Indicators are bounded words, not arbitrary substrings.
    assert formal.operational_family(
        "Tissue inventory", "A hanging ornament."
    ) == formal.INFORMATION
    assert formal.FAMILY_IDS == (
        "INFORMATION",
        "PROCEDURE",
        "TROUBLESHOOT",
    )
    assert formal.BLOCK_FAMILY_QUOTAS == {
        "A_form": 36,
        "F_search": 12,
        "A_hold": 12,
        "M_search": 12,
    }
    assert formal.SOURCE_MINIMUM_FAMILY_COUNTS == {
        "TRAIN": 48,
        "DEV": 24,
    }


def test_single_hmac_selection_is_deterministic_quota_exact_and_disjoint(
    source: formal.VerifiedSource,
) -> None:
    left = formal.select_private_cohorts(
        source, hmac_secret=b"a" * 32
    )
    right = formal.select_private_cohorts(
        source, hmac_secret=b"a" * 32
    )
    assert left.selection_sha256 == right.selection_sha256
    assert left.private_payload() == right.private_payload()
    selected = [
        row
        for block in left.blocks
        for row in block.items
    ]
    assert len({row.question.question_id for row in selected}) == len(
        selected
    )
    assert len(
        {row.question.normalized_query_sha256 for row in selected}
    ) == len(selected)
    assert len(
        {row.question.gold_document_id for row in selected}
    ) == len(selected)
    for block in left.blocks:
        assert {
            family: sum(row.family == family for row in block.items)
            for family in formal.FAMILY_IDS
        } == {
            family: formal.BLOCK_FAMILY_QUOTAS[block.block]
            for family in formal.FAMILY_IDS
        }


def test_hmac_order_is_not_label_dependent() -> None:
    title, text = _family_question_text(
        formal.INFORMATION, "label-independent-order"
    )
    candidates = tuple(f"doc-{index:02d}" for index in range(50))
    left = formal.VerifiedQuestion(
        question_id="same-question",
        question_title=title,
        question_text=text,
        document_ids=candidates,
        gold_document_id=candidates[0],
    )
    right = replace(left, gold_document_id=candidates[1])
    assert formal._selection_digest(
        b"h" * formal.HMAC_SECRET_BYTES,
        split="TRAIN",
        family=formal.INFORMATION,
        question=left,
    ) == formal._selection_digest(
        b"h" * formal.HMAC_SECRET_BYTES,
        split="TRAIN",
        family=formal.INFORMATION,
        question=right,
    )


def test_candidate_order_is_erased_and_shared_distractors_are_allowed(
    source: formal.VerifiedSource,
) -> None:
    reversed_source = _source(reverse_candidates=True)
    left = formal.select_private_cohorts(
        source, hmac_secret=b"b" * 32
    )
    right = formal.select_private_cohorts(
        reversed_source, hmac_secret=b"b" * 32
    )
    assert left.selection_sha256 == right.selection_sha256
    left_clusters = formal.build_search_clusters(
        source, left.block(formal.A_HOLD)
    )
    right_clusters = formal.build_search_clusters(
        reversed_source, right.block(formal.A_HOLD)
    )
    assert [
        row.corpus_sha256 for row in left_clusters
    ] == [row.corpus_sha256 for row in right_clusters]
    # Forty-nine distractors are shared by every item.  This is deliberately
    # accepted: there is no candidate-component disjointness gate.
    assert all(len(row.documents) == 58 for row in left_clusters)
    assert all(
        [row.document_id for row in cluster.documents]
        == sorted(row.document_id for row in cluster.documents)
        for cluster in left_clusters
    )


def test_public_action_projection_has_no_label_or_identity_channel(
    source: formal.VerifiedSource,
) -> None:
    selection = formal.select_private_cohorts(
        source, hmac_secret=b"c" * 32
    )
    cluster = formal.build_search_clusters(
        source, selection.block(formal.A_HOLD)
    )[0]
    item = cluster.items[0].selected.question
    projection = formal.public_action_projection(
        item, cluster.documents
    )
    assert set(projection) == {
        "documents",
        "question_text",
        "question_title",
    }
    assert all(
        set(value) == {"ordinal", "text", "title"}
        for value in projection["documents"]
    )
    serialized = json.dumps(projection, sort_keys=True)
    for forbidden in (
        "answer",
        "cluster",
        "document_id",
        "family",
        "gold",
        "qrel",
        "question_id",
        "source",
        "stage",
    ):
        assert f'"{forbidden}"' not in serialized


def test_promotion_reality_and_l5_use_exact_all_cluster_criteria() -> None:
    promotion = formal.compare_exact_rows(
        left_arm="E1",
        right_arm="E0",
        rows=_comparison_rows(),
    )
    assert promotion.mean_delta == 1
    assert promotion.one_sided_cluster_sign_tail == Fraction(1, 16)
    assert formal.promotion_criterion(promotion)
    assert formal.l5_criterion(promotion)
    assert formal.authorize_m_search(promotion) is not None

    failed = formal.compare_exact_rows(
        left_arm="E1",
        right_arm="E0",
        rows=_comparison_rows(failed_cluster=3),
    )
    assert failed.mean_delta > 0
    assert not formal.promotion_criterion(failed)
    assert formal.authorize_m_search(failed) is None

    e0_raw = formal.compare_exact_rows(
        left_arm="E0",
        right_arm="RAW",
        rows=_comparison_rows(),
    )
    e0_hippo = formal.compare_exact_rows(
        left_arm="E0",
        right_arm="HippoRAG",
        rows=_comparison_rows(),
    )
    assert formal.reality_criterion(e0_raw, e0_hippo)
    assert not formal.reality_criterion(
        e0_raw,
        formal.compare_exact_rows(
            left_arm="E0",
            right_arm="HippoRAG",
            rows=_comparison_rows(failed_cluster=2),
        ),
    )


def test_m_search_is_not_materialized_without_promotion_and_archives_separate(
    prepared: formal.PreparedStudy,
) -> None:
    assert prepared.prepromotion_private_payload()[
        "M_search_action_materialized"
    ] is False
    assert formal.M_SEARCH not in prepared.prepromotion_private_payload()[
        "stages"
    ]
    controller = formal.OneShotFormalController(prepared)
    result = controller.finalize(_hippo_results(prepared))
    # The synthetic action signatures make E1 fall back to E0, so A_hold
    # cannot promote and M_search must remain unopened.
    assert result.safe_terminal["A_hold"]["promotion_passed"] is False
    assert result.safe_terminal["M_search"][
        "actions_materialized_after_promotion"
    ] is False
    assert result.m_search is None
    assert controller.consumed
    with pytest.raises(formal.TechqaP1FormalError, match="replay"):
        controller.finalize(_hippo_results(prepared))

    private_text = json.dumps(
        result.private_archive, sort_keys=True
    )
    safe_text = json.dumps(result.safe_terminal, sort_keys=True)
    assert '"actions"' in private_text
    assert '"qrels"' in private_text
    assert "techqa-work-v1-" in private_text
    assert "techqa-work-v1-" not in safe_text
    assert "unique-000" not in safe_text
    assert result.safe_terminal[
        "item_query_document_qrel_action_values_published"
    ] is False
    assert result.safe_terminal["online_or_API_evaluator_call_count"] == 0


def test_official_hippo_exact_byte_binding_fails_closed(
    prepared: formal.PreparedStudy,
) -> None:
    results = list(_hippo_results(prepared))
    results[0] = replace(results[0], input_sha256="0" * 64)
    with pytest.raises(
        formal.TechqaP1FormalError,
        match="byte binding mismatch",
    ):
        formal.bind_official_hippo_results(
            prepared.a_hold, tuple(results)
        )

    valid = list(_hippo_results(prepared))
    first = valid[0]
    request = {
        row.work_id: row for row in prepared.hippo_requests
    }[first.work_id]
    ordinals = tuple(range(5))
    wrong_ids = tuple(
        str(request.documents[value + 1]["document_id"])
        for value in ordinals
    )
    valid[0] = replace(
        first,
        top5_ordinals=ordinals,
        top5_document_ids=wrong_ids,
    )
    with pytest.raises(
        formal.TechqaP1FormalError,
        match="mapping disagrees",
    ):
        formal.bind_official_hippo_results(
            prepared.a_hold, tuple(valid)
        )
