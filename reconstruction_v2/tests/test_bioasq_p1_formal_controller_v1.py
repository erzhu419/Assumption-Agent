from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import stat
from typing import Mapping, Sequence

import pytest

from assumption_agent.benchmarks import bioasq_p1_formal_controller_v1 as formal
from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core


SCORE_VECTORS = {
    "raw_ce": (8, 7, 6, 5, 4, 3, 2, 1),
    "focus_ce": (1, 2, 3, 4, 5, 6, 7, 8),
    "dense_base": (8, 7, 6, 5, 4, 3, 2, 1),
    "dense_support": (1, 3, 5, 7, 8, 6, 4, 2),
    "dense_contrast": (2, 4, 6, 8, 7, 5, 3, 1),
    "dense_coverage": (4, 8, 3, 7, 2, 6, 1, 5),
}
QUESTIONS = (
    "Is aspirin useful for condition alpha?",
    "What gene is involved in condition beta?",
    "List the drugs involved in condition gamma.",
    "Explain the mechanism for condition delta.",
)
A_FORM_GOLD = {
    core.B0_CLAIM: (4,),
    core.B1_ENTITY: (7,),
    core.B2_LIST: (4,),
    core.B3_ASPECT: (5,),
}


@pytest.fixture
def small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(formal, "CORPUS_SIZE", 8)
    monkeypatch.setattr(core, "CORPUS_SIZE", 8)
    monkeypatch.setattr(core, "LIST_DIVERSITY_CANDIDATE_PREFIX", 8)
    monkeypatch.setattr(core, "MIN_BUCKET_SUPPORT", 1)
    monkeypatch.setattr(core, "MIN_NET_POSITIVE_MARGIN_COUNT", 1)
    monkeypatch.setattr(
        formal,
        "BLOCK_COUNTS",
        {
            "A_form": 4,
            "F_search": 4,
            "A_hold": 4,
            "M_search": 4,
        },
    )
    monkeypatch.setattr(
        formal,
        "FAMILY_COUNTS",
        {
            "A_form": 1,
            "F_search": 1,
            "A_hold": 1,
            "M_search": 1,
        },
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _work(block: str, ordinal: int) -> str:
    return f"bioasq-work-v2-{_digest(f'{block}:{ordinal}')}"


def _corpus() -> formal.CorpusView:
    return formal.CorpusView.create(
        tuple(
            core.Passage(
                ordinal=ordinal,
                text=f"passage {ordinal} distinct token{ordinal}",
            )
            for ordinal in range(formal.CORPUS_SIZE)
        )
    )


def _block(block: str) -> formal.BlockView:
    return formal.BlockView.create(
        block,
        tuple(
            formal.FormalItemView(
                work_id=_work(block, ordinal),
                question_text=(
                    f"{QUESTIONS[ordinal]} Study block {block} item "
                    f"{ordinal}."
                ),
            )
            for ordinal in range(formal.BLOCK_COUNTS[block])
        ),
    )


class _Scorer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, ...]] = []
        self.seen_item_fields: set[str] = set()

    def score(
        self,
        corpus: formal.CorpusView,
        items: Sequence[formal.FormalItemView],
    ) -> Sequence[formal.CoordinateScoreRow]:
        self.events.append(f"coordinate:{len(items)}")
        self.calls.append(tuple(item.work_id for item in items))
        self.seen_item_fields.update(
            field.name for field in fields(items[0])
        )
        return tuple(
            formal.CoordinateScoreRow.create(
                item=item,
                corpus=corpus,
                score_vectors=SCORE_VECTORS,
            )
            for item in items
        )


class _Hippo:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, ...]] = []

    def retrieve(
        self,
        corpus: formal.CorpusView,
        items: Sequence[formal.FormalItemView],
    ) -> Sequence[formal.HippoResult]:
        self.events.append(f"hippo:{len(items)}")
        self.calls.append(tuple(item.work_id for item in items))
        return tuple(
            formal.HippoResult(
                work_id=item.work_id,
                normalized_query_sha256=hashlib.sha256(
                    item.question_text.encode("utf-8")
                ).hexdigest(),
                corpus_projection_sha256=corpus.projection_sha256,
                top5_ordinals=(0, 1, 2, 3, 4),
                receipt_sha256=_digest(f"hippo:{item.work_id}"),
            )
            for item in items
        )


class _Acquisition:
    def __init__(
        self,
        root: Path,
        events: list[str],
        *,
        force_no_promotion: bool = False,
    ) -> None:
        self.root = root
        self.events = events
        self.force_no_promotion = force_no_promotion
        self.corpus = _corpus()
        self.blocks = {
            block: _block(block) for block in formal.BLOCK_COUNTS
        }
        self.qrel_blocks: list[str] = []
        self.m_load_count = 0

    def claim_formal_attempt(
        self,
        formal_marker_sha256: str,
    ) -> formal.AcquisitionClaim:
        assert (self.root / formal.FORMAL_MARKER_FILENAME).is_file()
        self.events.append("claim")
        return formal.AcquisitionClaim.create(
            source_identity_commitment=_digest("source"),
            corpus_selection_commitment=_digest("corpus"),
            block_disjointness_commitment=_digest("disjoint"),
            source_qualification_commitment=_digest("qualification"),
        )

    def load_public_corpus(
        self,
        claim: formal.AcquisitionClaim,
    ) -> formal.CorpusView:
        assert isinstance(claim, formal.AcquisitionClaim)
        self.events.append("corpus")
        return self.corpus

    def load_label_free_block(
        self,
        block: str,
        authorization: Mapping[str, object] | None = None,
    ) -> formal.BlockView:
        if block == "M_search":
            self.m_load_count += 1
            assert authorization is not None
            body = dict(authorization)
            claimed = body.pop("self_sha256")
            assert claimed == formal.stable_hash(body)
            assert (
                self.root / formal.PROMOTION_AUTHORIZATION_FILENAME
            ).is_file()
        else:
            assert authorization is None
        self.events.append(f"load:{block}")
        return self.blocks[block]

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> formal.QrelPack:
        assert block != "F_search"
        assert custody_path.is_file()
        assert stat.S_IMODE(custody_path.stat().st_mode) == 0o400
        body = dict(sealed_action_archive)
        claimed = body.pop("self_sha256")
        assert claimed == formal.stable_hash(body)
        self.events.append(f"qrel:{block}")
        self.qrel_blocks.append(block)

        action_rows = {
            str(row["work_id"]): row
            for row in sealed_action_archive["rows"]  # type: ignore[index]
        }
        qrels: list[formal.QrelRow] = []
        for family, item in zip(
            formal.FAMILIES,
            self.blocks[block].items,
        ):
            if block == "A_form":
                bucket = core.predict_question_structure(
                    item.question_text
                ).predicted_bucket
                gold = A_FORM_GOLD[bucket]
            else:
                arms = action_rows[item.work_id]["arms"]
                if self.force_no_promotion and block == "A_hold":
                    gold = (int(arms["E0"][0]),)
                else:
                    excluded = set(arms["E0"]) | set(arms["RAW"]) | set(
                        arms["HippoRAG"]
                    )
                    candidate = next(
                        ordinal
                        for ordinal in arms["E1"]
                        if ordinal not in excluded
                    )
                    gold = (int(candidate),)
            qrels.append(
                formal.QrelRow(
                    work_id=item.work_id,
                    family=family,
                    gold_ordinals=gold,
                    corpus_projection_sha256=(
                        self.corpus.projection_sha256
                    ),
                )
            )
        return formal.QrelPack.create(
            block=block,
            action_archive_sha256=str(
                sealed_action_archive["self_sha256"]
            ),
            rows=qrels,
        )


def _run(
    tmp_path: Path,
    *,
    force_no_promotion: bool = False,
):
    events: list[str] = []
    acquisition = _Acquisition(
        tmp_path,
        events,
        force_no_promotion=force_no_promotion,
    )
    scorer = _Scorer(events)
    hippo = _Hippo(events)
    terminal = formal.run_formal_controller(
        work_root=tmp_path,
        execution_binding_sha256=_digest("execution"),
        acquisition=acquisition,
        coordinate_scorer=scorer,
        hippo_runner=hippo,
    )
    return terminal, acquisition, scorer, hippo, events


def test_frozen_contract_and_public_projection_are_exact() -> None:
    assert formal.STUDY_ID == (
        "BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1"
    )
    assert formal.FAMILIES == ("yesno", "factoid", "list", "summary")
    assert formal.BLOCK_COUNTS == {
        "A_form": 96,
        "F_search": 32,
        "A_hold": 48,
        "M_search": 48,
    }
    assert formal.FAMILY_COUNTS == {
        "A_form": 24,
        "F_search": 8,
        "A_hold": 12,
        "M_search": 12,
    }
    assert tuple(field.name for field in fields(formal.FormalItemView)) == (
        "work_id",
        "question_text",
    )
    assert core.RECIPE_IDS == (
        "claim_polarity_balanced_evidence_set",
        "entity_focused_evidence_set",
        "list_redundancy_controlled_evidence_set",
        "multi_aspect_coverage_evidence_set",
        "global_raw_dense_reciprocal_rank_fusion",
    )


def test_set_utility_uses_exact_half_even_rounding(
    small_contract: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert formal.set_recall_first_rr_utility(
        (0, 1, 2, 3, 4),
        (0,),
    ) == 600_000
    assert formal.set_recall_first_rr_utility(
        (7, 6, 5, 4, 3),
        (4,),
    ) == 375_000
    # One of four qrels is retrieved at rank five.
    assert formal.set_recall_first_rr_utility(
        (0, 1, 2, 3, 4),
        tuple(range(5, 8)) + (4,),
    ) == 135_000
    # 1/64 recall is 4687.5, which rounds to the even integer 4688.
    monkeypatch.setattr(formal, "CORPUS_SIZE", 128)
    monkey_gold = (4, *range(5, 68))
    assert formal.set_recall_first_rr_utility(
        (0, 1, 2, 3, 4),
        monkey_gold,
    ) == 64_688


def test_exact_magnitude_preserving_tail() -> None:
    comparison = formal.compare_paired_integer_utility(
        (9, 8, 7, 6),
        (0, 0, 0, 0),
    )
    assert comparison.net_utility == 30
    assert comparison.positive_count == 4
    assert (
        comparison.one_sided_exact_magnitude_preserving_tail
        == Fraction(1, 16)
    )
    ties = formal.compare_paired_integer_utility((1, 2), (1, 2))
    assert ties.one_sided_exact_magnitude_preserving_tail == 1


def test_full_promoted_lifecycle_is_late_qrel_private_and_two_batch(
    tmp_path: Path,
    small_contract: None,
) -> None:
    terminal, acquisition, scorer, hippo, events = _run(tmp_path)
    assert terminal["status"] == (
        "terminal_complete_after_A_hold_promotion_and_M_search"
    )
    assert terminal["A_hold"]["promotion_passed"] is True
    assert terminal["A_hold"]["reality_primary_passed"] is True
    assert terminal["A_hold"]["stable_family_count"] == 4
    assert terminal["M_search"]["opened_after_promotion"] is True
    assert terminal["M_search"]["L5_passed"] is True
    assert acquisition.qrel_blocks == ["A_form", "A_hold", "M_search"]
    assert acquisition.m_load_count == 1
    assert [len(call) for call in scorer.calls] == [12, 4]
    assert [len(call) for call in hippo.calls] == [4, 4]
    assert scorer.seen_item_fields == {"work_id", "question_text"}
    counts = terminal["formal_protocol_call_counts"]
    assert counts["coordinate_score_invocations"] == 2
    assert counts["official_HippoRAG_retrieve_invocations"] == 2
    assert counts["E1_fit_invocations"] == 1
    assert counts["qrel_release_invocations"] == 3
    assert events.index("qrel:A_form") > events.index("coordinate:12")
    assert events.index("qrel:A_hold") > events.index("qrel:A_form")
    assert events.index("load:M_search") > events.index("qrel:A_hold")

    for path in tmp_path.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
    public_terminal = json.loads(
        (tmp_path / formal.FORMAL_TERMINAL_FILENAME).read_text("ascii")
    )
    serialized = json.dumps(public_terminal, sort_keys=True)
    assert "question_text" not in serialized
    assert "gold_ordinals" not in serialized
    assert '"work_id"' not in serialized
    assert "top5_ordinals" not in serialized
    assert public_terminal["online_or_API_evaluator_calls"] == 0
    assert public_terminal[formal.NO_CHANGE_COUNT_KEY] == 0


def test_failed_promotion_leaves_m_search_unopened(
    tmp_path: Path,
    small_contract: None,
) -> None:
    terminal, acquisition, scorer, hippo, _events = _run(
        tmp_path,
        force_no_promotion=True,
    )
    assert terminal["status"] == (
        "terminal_A_hold_E1_not_promoted_M_search_unopened"
    )
    assert terminal["A_hold"]["promotion_passed"] is False
    assert terminal["M_search"] == {
        "L5_E1_minus_E0": None,
        "L5_passed": None,
        "opened_after_promotion": False,
    }
    assert acquisition.m_load_count == 0
    assert acquisition.qrel_blocks == ["A_form", "A_hold"]
    assert [len(call) for call in scorer.calls] == [12]
    assert [len(call) for call in hippo.calls] == [4]
    assert not (
        tmp_path / formal.PROMOTION_AUTHORIZATION_FILENAME
    ).exists()
    assert not (tmp_path / "M_search.actions.private.json").exists()


def test_qrel_rows_reject_empty_duplicate_and_unsorted_sets(
    small_contract: None,
) -> None:
    common = {
        "work_id": _work("A_hold", 0),
        "family": "yesno",
        "corpus_projection_sha256": _digest("corpus"),
    }
    with pytest.raises(formal.BioasqP1FormalControllerError):
        formal.QrelRow(gold_ordinals=(), **common)
    with pytest.raises(formal.BioasqP1FormalControllerError):
        formal.QrelRow(gold_ordinals=(1, 1), **common)
    with pytest.raises(formal.BioasqP1FormalControllerError):
        formal.QrelRow(gold_ordinals=(2, 1), **common)


def test_one_shot_marker_prevents_replay(
    tmp_path: Path,
    small_contract: None,
) -> None:
    _run(tmp_path)
    with pytest.raises(
        formal.BioasqP1FormalControllerError,
        match="one-shot archive already exists",
    ):
        _run(tmp_path)
