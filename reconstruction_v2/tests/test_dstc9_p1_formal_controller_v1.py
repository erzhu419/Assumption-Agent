from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import stat
from typing import Mapping, Sequence

import pytest

from assumption_agent.benchmarks import dstc9_p1_formal_controller_v1 as formal
from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as core


SUCCESS_VECTORS = {
    "global_ce": (6, 7, 3, 0, 2, 5, 1, 4),
    "last_turn_ce": (4, 3, 2, 7, 1, 5, 0, 6),
    "minilm": (2, 6, 0, 4, 7, 3, 5, 1),
    "entity": (1, 5, 0, 2, 3, 4, 6, 7),
    "title": (6, 2, 4, 1, 3, 5, 7, 0),
    "body": (0, 7, 5, 3, 6, 2, 1, 4),
}
FLAT_VECTORS = {
    name: tuple(0 for _ in range(8)) for name in core.SCORE_NAMES
}
GOLD_ORDINAL = 4


@pytest.fixture
def small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(formal, "CORPUS_SIZE", 8)
    monkeypatch.setattr(
        formal,
        "BLOCK_COUNTS",
        {
            "A_form": 12,
            "F_search": 4,
            "A_hold": 4,
            "M_search": 4,
        },
    )
    monkeypatch.setattr(
        formal,
        "FAMILY_COUNTS",
        {
            "A_form": 3,
            "F_search": 1,
            "A_hold": 1,
            "M_search": 1,
        },
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _work(block: str, ordinal: int) -> str:
    return f"dstc9-work-v1-{_digest(f'{block}:{ordinal}')}"


def _corpus() -> formal.CorpusView:
    return formal.CorpusView.create(
        tuple(
            core.KnowledgeSnippet(
                ordinal=ordinal,
                entity_name=f"entity {ordinal}",
                title=f"title {ordinal}",
                body=f"body {ordinal}",
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
                block=block,
                history=(
                    core.DialogueTurn(
                        "U",
                        f"Unique public {block} query {ordinal}",
                    ),
                ),
            )
            for ordinal in range(formal.BLOCK_COUNTS[block])
        ),
    )


class _Predictor:
    def __init__(
        self,
        commitment: str,
        events: list[str],
        *,
        broken: bool = False,
    ) -> None:
        self.commitment = commitment
        self.events = events
        self.broken = broken
        self.seen_fields: set[str] = set()

    def predict(
        self,
        items: Sequence[formal.FormalItemView],
    ) -> Sequence[formal.BucketPrediction]:
        block = items[0].block
        self.events.append(f"predict:{block}")
        self.seen_fields.update(field.name for field in fields(items[0]))
        if self.broken:
            return ()
        return tuple(
            formal.BucketPrediction.create(
                item=item,
                predicted_bucket=0,
                predictor_commitment=self.commitment,
            )
            for item in items
        )


class _Scorer:
    def __init__(
        self,
        vectors: Mapping[str, tuple[int, ...]],
        events: list[str],
    ) -> None:
        self.vectors = vectors
        self.events = events
        self.seen_item_fields: set[str] = set()
        self.corpus_calls: list[str] = []

    def score(
        self,
        corpus: formal.CorpusView,
        items: Sequence[formal.FormalItemView],
    ) -> Sequence[formal.CoordinateScoreRow]:
        block = items[0].block
        self.events.append(f"coordinate:{block}")
        self.corpus_calls.append(corpus.view_sha256)
        self.seen_item_fields.update(
            field.name for field in fields(items[0])
        )
        return tuple(
            formal.CoordinateScoreRow.create(
                item=item,
                corpus=corpus,
                score_vectors=self.vectors,
            )
            for item in items
        )


class _Hippo:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.corpus_calls: list[str] = []

    def retrieve(
        self,
        corpus: formal.CorpusView,
        items: Sequence[formal.FormalItemView],
    ) -> Sequence[formal.HippoResult]:
        block = items[0].block
        self.events.append(f"hippo:{block}")
        self.corpus_calls.append(corpus.view_sha256)
        return tuple(
            formal.HippoResult(
                work_id=item.work_id,
                block=item.block,
                normalized_query_sha256=core.normalized_query_sha256(
                    item.history
                ),
                corpus_projection_sha256=corpus.projection_sha256,
                top5_ordinals=(0, 1, 2, 3, 5),
                receipt_sha256=_digest(f"hippo:{item.work_id}"),
            )
            for item in items
        )


class _Acquisition:
    def __init__(
        self,
        root: Path,
        predictor_commitment: str,
        events: list[str],
        *,
        forbid_m: bool = False,
    ) -> None:
        self.root = root
        self.predictor_commitment = predictor_commitment
        self.events = events
        self.forbid_m = forbid_m
        self.corpus = _corpus()
        self.blocks = {
            block: _block(block) for block in formal.BLOCK_COUNTS
        }
        self.qrel_blocks: list[str] = []

    def claim_formal_attempt(
        self,
        formal_marker_sha256: str,
    ) -> formal.AcquisitionClaim:
        assert (
            self.root / formal.FORMAL_MARKER_FILENAME
        ).is_file()
        self.events.append("claim")
        return formal.AcquisitionClaim.create(
            source_identity_commitment=_digest("source"),
            corpus_selection_commitment=_digest("selection"),
            block_disjointness_commitment=_digest("disjoint"),
            query_only_predictor_commitment=(
                self.predictor_commitment
            ),
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
            if self.forbid_m:
                raise AssertionError("M_search materialized without promotion")
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
        assert sealed_action_archive["self_sha256"] == formal.stable_hash(
            {
                key: value
                for key, value in sealed_action_archive.items()
                if key != "self_sha256"
            }
        )
        self.events.append(f"qrel:{block}")
        self.qrel_blocks.append(block)
        rows = tuple(
            formal.QrelRow(
                work_id=item.work_id,
                family=formal.FAMILIES[index % len(formal.FAMILIES)],
                gold_ordinal=GOLD_ORDINAL,
                corpus_projection_sha256=(
                    self.corpus.projection_sha256
                ),
            )
            for index, item in enumerate(self.blocks[block].items)
        )
        return formal.QrelPack.create(
            block=block,
            action_archive_sha256=str(
                sealed_action_archive["self_sha256"]
            ),
            rows=rows,
        )


def _run(
    tmp_path: Path,
    *,
    vectors: Mapping[str, tuple[int, ...]],
    forbid_m: bool = False,
    broken_predictor: bool = False,
):
    events: list[str] = []
    predictor_commitment = _digest("query-only-predictor")
    acquisition = _Acquisition(
        tmp_path,
        predictor_commitment,
        events,
        forbid_m=forbid_m,
    )
    predictor = _Predictor(
        predictor_commitment,
        events,
        broken=broken_predictor,
    )
    scorer = _Scorer(vectors, events)
    hippo = _Hippo(events)
    result = formal.run_formal_controller(
        work_root=tmp_path,
        execution_binding_sha256=_digest("execution"),
        acquisition=acquisition,
        predictor=predictor,
        coordinate_scorer=scorer,
        hippo_runner=hippo,
    )
    return result, acquisition, predictor, scorer, hippo, events


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for member in value.values():
            result.update(_all_keys(member))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for member in value:
            result.update(_all_keys(member))
        return result
    return set()


def test_frozen_formal_contract_and_exact_sign_flip() -> None:
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
    assert formal.CORPUS_SIZE == 2_900
    comparison = formal.compare_paired_integer_utility(
        (120, 90, 80, 75),
        (0, 0, 0, 0),
    )
    assert comparison.net_utility == 365
    assert comparison.one_sided_exact_magnitude_preserving_tail == (
        Fraction(1, 16)
    )
    assert formal.singleton_recall_rr_utility((4, 0, 1, 2, 3), 4) == 120
    assert formal.singleton_recall_rr_utility((0, 4, 1, 2, 3), 4) == 90
    assert formal.singleton_recall_rr_utility((0, 1, 2, 3, 5), 4) == 0


def test_success_lifecycle_is_late_qrel_label_free_and_private(
    tmp_path: Path,
    small_contract: None,
) -> None:
    (
        terminal,
        acquisition,
        predictor,
        scorer,
        hippo,
        events,
    ) = _run(tmp_path, vectors=SUCCESS_VECTORS)
    assert terminal["status"] == (
        "terminal_complete_after_A_hold_promotion_and_M_search"
    )
    assert terminal["A_hold"]["promotion_passed"] is True
    assert terminal["A_hold"]["reality_primary_passed"] is True
    assert terminal["M_search"]["L5_passed"] is True
    assert terminal["M_search"]["opened_after_promotion"] is True
    assert terminal["F_search"] == {
        "label_or_qrel_open_count": 0,
        "model_fit_or_update_count": 0,
        "private_behavior_archive_sha256": terminal[
            "F_search"
        ]["private_behavior_archive_sha256"],
    }
    assert acquisition.qrel_blocks == [
        "A_form",
        "A_hold",
        "M_search",
    ]
    assert "qrel:F_search" not in events
    assert events.index("qrel:A_form") > events.index(
        "coordinate:A_form"
    )
    assert events.index("qrel:A_hold") > events.index(
        "hippo:A_hold"
    )
    assert events.index("load:M_search") > events.index(
        "qrel:A_hold"
    )
    assert events.index("qrel:M_search") > events.index(
        "hippo:M_search"
    )
    assert predictor.seen_fields == {"work_id", "block", "history"}
    assert scorer.seen_item_fields == {"work_id", "block", "history"}
    assert len(set(scorer.corpus_calls)) == 1
    assert len(set(hippo.corpus_calls)) == 1

    action = json.loads(
        (tmp_path / "A_hold.actions.private.json").read_text("ascii")
    )
    forbidden = {
        "answer",
        "doc_id",
        "document_id",
        "domain",
        "entity_id",
        "family",
        "gold",
        "gold_ordinal",
        "label",
        "qrel",
        "qrels",
        "response",
        "split",
        "target",
        "utility",
    }
    assert not (_all_keys(action) & forbidden)
    assert action["label_bearing_input_count"] == 0
    assert all(
        row["prediction"]["query_only_input_contract"] is True
        for row in action["rows"]
    )

    terminal_raw = formal.canonical_bytes(terminal).decode("ascii")
    assert "dstc9-work-v1-" not in terminal_raw
    assert '"gold_ordinal"' not in terminal_raw
    assert '"normalized_query_sha256"' not in terminal_raw
    for path in tmp_path.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400


def test_failed_promotion_never_materializes_M_search(
    tmp_path: Path,
    small_contract: None,
) -> None:
    terminal, acquisition, _p, _s, _h, events = _run(
        tmp_path,
        vectors=FLAT_VECTORS,
        forbid_m=True,
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
    assert acquisition.qrel_blocks == ["A_form", "A_hold"]
    assert "load:M_search" not in events
    assert not (
        tmp_path / formal.PROMOTION_AUTHORIZATION_FILENAME
    ).exists()
    assert not (tmp_path / "M_search.actions.private.json").exists()
    assert not (tmp_path / "M_search.qrels.private.json").exists()


def test_failure_terminal_hashes_error_and_does_not_leak(
    tmp_path: Path,
    small_contract: None,
) -> None:
    with pytest.raises(
        formal.Dstc9P1FormalControllerError,
        match="prediction coverage",
    ):
        _run(
            tmp_path,
            vectors=SUCCESS_VECTORS,
            broken_predictor=True,
        )
    path = tmp_path / formal.FORMAL_TERMINAL_FILENAME
    terminal = json.loads(path.read_text("ascii"))
    assert terminal["status"] == "terminal_formal_failure_no_retry"
    assert terminal["failure_stage"] == "form_and_seal_A_form_actions"
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    raw = formal.canonical_bytes(terminal).decode("ascii")
    assert "Unique public" not in raw
    assert "dstc9-work-v1-" not in raw


def test_cross_block_normalized_query_overlap_fails_before_action(
    tmp_path: Path,
    small_contract: None,
) -> None:
    events: list[str] = []
    predictor_commitment = _digest("query-only-predictor")
    acquisition = _Acquisition(
        tmp_path,
        predictor_commitment,
        events,
    )
    aform_first = acquisition.blocks["A_form"].items[0]
    f_items = list(acquisition.blocks["F_search"].items)
    f_items[0] = formal.FormalItemView(
        work_id=f_items[0].work_id,
        block="F_search",
        history=aform_first.history,
    )
    acquisition.blocks["F_search"] = formal.BlockView.create(
        "F_search",
        f_items,
    )
    predictor = _Predictor(predictor_commitment, events)
    with pytest.raises(
        formal.Dstc9P1FormalControllerError,
        match="not block-disjoint",
    ):
        formal.run_formal_controller(
            work_root=tmp_path,
            execution_binding_sha256=_digest("execution"),
            acquisition=acquisition,
            predictor=predictor,
            coordinate_scorer=_Scorer(SUCCESS_VECTORS, events),
            hippo_runner=_Hippo(events),
        )
    assert "predict:A_form" not in events
    terminal = json.loads(
        (tmp_path / formal.FORMAL_TERMINAL_FILENAME).read_text("ascii")
    )
    assert terminal["failure_stage"] == "load_A_hold_label_free"
