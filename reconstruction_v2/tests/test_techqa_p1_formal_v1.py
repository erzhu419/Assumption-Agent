from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from assumption_agent.benchmarks import techqa_p1_formal_v1 as formal
from assumption_agent.benchmarks import (
    techqa_p1_official_hipporag_v1 as adapter,
)
from replication_runtime.averitec_p1_official_v1 import worker as inner


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


@pytest.fixture
def secure_tmp_path() -> Path:
    root = Path(tempfile.mkdtemp(prefix="techqa-formal-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    @staticmethod
    def vcount() -> int:
        return 9

    @staticmethod
    def ecount() -> int:
        return 11


class _Core:
    def __init__(self, index_root: Path) -> None:
        self.graph = _Graph()
        self.index_root = index_root
        self.documents: list[str] = []

    def index(self, documents: list[str]) -> None:
        self.documents = list(documents)

    def retrieve(
        self,
        queries: list[str],
        *,
        num_to_retrieve: int,
    ) -> list[_Solution]:
        assert num_to_retrieve == len(self.documents)
        (self.index_root / "query.cache").write_bytes(b"formal fake cache")
        return [
            _Solution(
                docs=list(self.documents),
                doc_scores=[
                    float(len(self.documents) - ordinal)
                    for ordinal in range(len(self.documents))
                ],
            )
            for _query in queries
        ]


def _cuda_phase() -> dict[str, object]:
    return {
        "cuda_allocation_and_synchronize_succeeded": True,
        "logical_cuda_current_device": 0,
        "physical_visible_gpu_binding": "0",
        "torch_cuda_is_available": True,
        "visible_cuda_device_count": 1,
    }


class _FakeInner:
    def __call__(
        self,
        *,
        private_input: dict[str, object],
        index_root: Path,
    ) -> dict[str, object]:
        assert not index_root.exists()
        index_root.mkdir(mode=0o700)
        (index_root / "frozen.index").write_bytes(b"formal fake index")
        return inner.retrieve_with_core(
            core=_Core(index_root),
            private_input=private_input,
            index_root=index_root,
            cuda_receipt={
                "post_inference": _cuda_phase(),
                "pre_inference": _cuda_phase(),
            },
            observed_process_thread_peak=2,
        )


def _execute_hippo_cluster_runs(
    prepared: formal.PreparedStudy,
    *,
    work_root: Path,
) -> tuple[formal.OfficialHippoClusterRun, ...]:
    work_root.mkdir(mode=0o700)
    values: list[formal.OfficialHippoClusterRun] = []
    for request in prepared.hippo_cluster_requests:
        output = adapter.execute_cluster_once(
            request.adapter_input,
            work_root=work_root / f"cluster-{request.cluster_index}",
            inner_runner=_FakeInner(),
        )
        values.append(
            formal.OfficialHippoClusterRun(
                adapter_input=request.adapter_input,
                safe_output=output,
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
    for block in left.blocks:
        assert {
            family: sum(row.family == family for row in block.items)
            for family in formal.FAMILY_IDS
        } == {
            family: formal.BLOCK_FAMILY_QUOTAS[block.block]
            for family in formal.FAMILY_IDS
        }


def test_source_backed_selection_delegates_to_question_only_selector(
    source: formal.VerifiedSource,
) -> None:
    secret = b"q" * formal.HMAC_SECRET_BYTES
    delegated = formal.select_private_cohorts(
        source, hmac_secret=secret
    )
    direct = formal.select_question_cohorts(
        tuple(reversed(source.training_questions)),
        tuple(reversed(source.dev_questions)),
        hmac_secret=secret,
    )
    assert direct == delegated
    assert direct.private_payload() == delegated.private_payload()

    with pytest.raises(
        formal.TechqaP1FormalError,
        match="not unique across verified splits",
    ):
        formal.select_question_cohorts(
            source.training_questions,
            source.dev_questions + (source.training_questions[0],),
            hmac_secret=secret,
        )
    with pytest.raises(
        formal.TechqaP1FormalError,
        match="not a sequence",
    ):
        formal.select_question_cohorts(
            iter(source.training_questions),  # type: ignore[arg-type]
            source.dev_questions,
            hmac_secret=secret,
        )


def test_shared_gold_document_is_not_an_artificial_selection_gate(
    source: formal.VerifiedSource,
) -> None:
    training = list(source.training_questions)
    positions = [
        index
        for index, row in enumerate(training)
        if formal.operational_family(
            row.question_title, row.question_text
        )
        == formal.INFORMATION
    ][:2]
    for position in positions:
        training[position] = replace(
            training[position], gold_document_id="shared-00"
        )
    selection = formal.select_private_cohorts(
        replace(source, training_questions=tuple(training)),
        hmac_secret=b"g" * 32,
    )
    selected_gold = [
        row.question.gold_document_id
        for block in selection.blocks
        for row in block.items
    ]
    assert selected_gold.count("shared-00") == 2


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
        [
            row.private_payload()["serialized_sha256"]
            for row in cluster.documents
        ]
        == sorted(
            row.private_payload()["serialized_sha256"]
            for row in cluster.documents
        )
        for cluster in left_clusters
    )
    assert all(
        [row.selected.work_id for row in cluster.items]
        == sorted(row.selected.work_id for row in cluster.items)
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
    secure_tmp_path: Path,
) -> None:
    assert prepared.prepromotion_private_payload()[
        "M_search_action_materialized"
    ] is False
    assert formal.M_SEARCH not in prepared.prepromotion_private_payload()[
        "stages"
    ]
    controller = formal.OneShotFormalController(prepared)
    runs = _execute_hippo_cluster_runs(
        prepared, work_root=secure_tmp_path / "official"
    )
    result = controller.finalize(runs)
    # The synthetic action signatures make E1 fall back to E0, so A_hold
    # cannot promote and M_search must remain unopened.
    assert result.safe_terminal["A_hold"]["promotion_passed"] is False
    assert result.safe_terminal["M_search"][
        "actions_materialized_after_promotion"
    ] is False
    assert result.m_search is None
    assert controller.consumed
    with pytest.raises(formal.TechqaP1FormalError, match="replay"):
        controller.finalize(runs)

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
    assert result.safe_terminal[
        "cohort_question_and_normalized_query_disjoint"
    ] is True
    assert result.safe_terminal[
        "cohort_gold_document_disjoint"
    ] is False
    assert result.safe_terminal[
        "shared_corpus_and_gold_overlap_allowed"
    ] is True
    assert result.safe_terminal["M_search_untouched_scope"] == (
        "query_and_action_not_document_disjoint"
    )
    assert result.safe_terminal["online_or_API_evaluator_call_count"] == 0


def test_official_hippo_cluster_bridge_exact_coverage_and_mapping(
    prepared: formal.PreparedStudy,
    secure_tmp_path: Path,
) -> None:
    runs = _execute_hippo_cluster_runs(
        prepared, work_root=secure_tmp_path / "official"
    )
    assert len(runs) == 4
    bound = formal.bind_official_hippo_cluster_runs(
        prepared.a_hold, runs
    )
    assert set(bound) == {
        row.work_id for row in prepared.a_hold.actions
    }
    for request in prepared.hippo_cluster_requests:
        assert len(request.query_bindings) == 9
        for binding in request.query_bindings:
            row = bound[binding.work_id]
            assert row.cluster_index == request.cluster_index
            assert row.query_ordinal == binding.query_ordinal
            assert row.query_bytes_sha256 == binding.query_bytes_sha256
            assert row.cluster_request_sha256 == request.request_sha256

    with pytest.raises(
        formal.TechqaP1FormalError,
        match="coverage",
    ):
        formal.bind_official_hippo_cluster_runs(
            prepared.a_hold, runs[:-1]
        )


def test_official_hippo_cluster_input_and_receipt_mismatch_fail_closed(
    prepared: formal.PreparedStudy,
    source: formal.VerifiedSource,
    secure_tmp_path: Path,
) -> None:
    runs = list(
        _execute_hippo_cluster_runs(
            prepared, work_root=secure_tmp_path / "official"
        )
    )
    alternate = formal.prepare_formal_study(
        source, hmac_secret=b"t" * formal.HMAC_SECRET_BYTES
    )
    runs[0] = _execute_hippo_cluster_runs(
        alternate, work_root=secure_tmp_path / "alternate"
    )[0]
    with pytest.raises(
        formal.TechqaP1FormalError,
        match="exact adapter input mismatch",
    ):
        formal.bind_official_hippo_cluster_runs(
            prepared.a_hold, tuple(runs)
        )

    tampered_output = copy.deepcopy(
        _execute_hippo_cluster_runs(
            prepared, work_root=secure_tmp_path / "second"
        )[0].safe_output
    )
    tampered_output["attempt_marker_self_sha256"] = "0" * 64
    body = dict(tampered_output)
    body.pop("self_sha256")
    tampered_output["self_sha256"] = adapter.stable_hash(body)
    with pytest.raises(
        formal.TechqaP1FormalError,
        match="validation failed",
    ):
        formal.OfficialHippoClusterRun(
            adapter_input=prepared.hippo_cluster_requests[0].adapter_input,
            safe_output=tampered_output,
        )


def test_no_bare_top5_finalizer_and_cluster_text_is_stored_once(
    prepared: formal.PreparedStudy,
) -> None:
    assert not hasattr(formal, "OfficialHippoResult")
    with pytest.raises(
        formal.TechqaP1FormalError,
        match="coverage",
    ):
        formal.OneShotFormalController(prepared).finalize(
            ({"top5_ordinals": [0, 1, 2, 3, 4]},)  # type: ignore[arg-type]
        )
    assert len(prepared.hippo_cluster_requests) == 4
    assert all(
        len(request.adapter_input["queries"]) == 9
        for request in prepared.hippo_cluster_requests
    )
    assert sum(
        len(request.adapter_input["documents"])
        for request in prepared.hippo_cluster_requests
    ) == sum(
        len(cluster.documents) for cluster in prepared.a_hold.clusters
    )
    assert all(
        "hippo_request" not in action.private_payload()
        for action in prepared.a_hold.actions
    )
