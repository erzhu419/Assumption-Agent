from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from assumption_agent.benchmarks import hotpot_recursive_l4_v1 as l4
from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
)


PROJECT = Path(__file__).resolve().parents[1]


def _program(title: int, expansion: str = "none") -> TypedRetrievalProgram:
    return TypedRetrievalProgram(
        seed_algorithm="bm25",
        title_weight=title,
        text_weight=1,
        expansion_mode=expansion,
        expansion_weight=0 if expansion == "none" else 1,
    )


def _item(index: int) -> l4.RecursiveItem:
    corpus = tuple(
        RetrievalParagraph(
            idx=ordinal,
            title=f"title-{index}-{ordinal}",
            text=f"paragraph-{index}-{ordinal}",
        )
        for ordinal in range(7)
    )
    return l4.RecursiveItem(
        item_id=f"private-{index}",
        question=f"question-{index}",
        corpus=corpus,
        support_indices=(5, 6),
        row_commitment_sha256=f"{index + 1:064x}",
    )


def test_q_selection_excludes_p_equivalent_behavior_and_selects_complement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _program(1)
    equivalent = _program(2)
    complement = _program(3)
    distractor = _program(4)
    rankings = {
        p.program_hash: (5, 0, 1, 2, 3),
        equivalent.program_hash: (5, 0, 1, 2, 3),
        complement.program_hash: (6, 4, 0, 1, 2),
        distractor.program_hash: (4, 0, 1, 2, 3),
    }

    def ranking(program: TypedRetrievalProgram, _item: Any) -> tuple[int, ...]:
        return rankings[program.program_hash]

    monkeypatch.setattr(l4, "_ranking", ranking)
    monkeypatch.setattr(
        l4, "enumerate_programs", lambda: iter((equivalent, complement, distractor))
    )
    winner, candidates, _p_behavior = l4._select_q(
        p_program=p,
        items=tuple(_item(index) for index in range(8)),
    )
    assert winner.program.program_hash == complement.program_hash
    assert {row.program.program_hash for row in candidates} == {
        complement.program_hash,
        distractor.program_hash,
    }
    assert winner.combined_hits == 16
    assert winner.novelty_added == 8
    assert winner.retained_added == 8
    assert winner.invalid_count == 0


def test_l4_score_reports_retention_novelty_forgetting_and_official_context() -> None:
    items = (_item(0), _item(1))
    direct: dict[tuple[int, str], tuple[int, ...]] = {}
    for ordinal in range(2):
        direct[(ordinal, "canonical_RAW")] = (0, 1, 2, 3, 4)
        direct[(ordinal, "retained_P")] = (5, 0, 1, 2, 3)
        direct[(ordinal, "novel_Q")] = (6, 4, 0, 1, 2)
        direct[(ordinal, "official_HippoRAG")] = (5, 6, 0, 1, 2)
    result = l4._score(items=items, direct=direct)
    metrics = result["arm_metrics"]
    assert metrics["canonical_RAW"]["support_hit_count"] == 0
    assert metrics["retained_P"]["support_hit_count"] == 2
    assert metrics["novel_Q"]["support_hit_count"] == 2
    assert metrics["P_plus_Q_RRF"]["support_hit_count"] == 4
    assert metrics["official_HippoRAG"]["support_hit_count"] == 4
    assert result["retained_P_contribution"]["net_support_hit_count"] == 2
    assert result["novel_Q_contribution"]["net_support_hit_count"] == 2
    assert result["forgetting"]["support_hit_count_P_lost_in_P_plus_Q"] == 0
    assert result["P_plus_Q_minus_official_HippoRAG"][
        "net_support_hit_count"
    ] == 0
    assert result["disposition"]["retained_improvement_observed"] is True
    assert result["disposition"]["novel_improvement_observed"] is True
    assert result["disposition"][
        "positive_net_on_fixed_cohort_vs_official_HippoRAG"
    ] is False
    assert result["disposition"]["statistical_superiority_claim"] is False
    assert result["disposition"]["family_out_claim_for_P_plus_Q"] is False
    assert result["disposition"]["compute_budget_equivalence_claim"] is False


def test_positive_p_lineage_recomputes_committed_m1_chain() -> None:
    p_program, binding = l4._load_p(
        project=PROJECT,
        formation_path=(
            PROJECT
            / "manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json"
        ),
        program_path=(
            PROJECT
            / "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json"
        ),
    )
    lineage = l4._load_positive_p_lineage(
        m1_freeze_path=PROJECT / "manifests/musique_recursive_study_m1_pre_run_freeze_v1.json",
        m1_report_path=PROJECT / "manifests/musique_recursive_study_m1_aggregate_report_v1.json",
        p_binding=binding,
    )
    assert lineage["program_hash"] == p_program.program_hash
    assert lineage["disposition"] == "promote_P_to_retained_generation_one"


def test_q_formation_marker_precedes_private_open_and_blocks_tampered_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "q-formation.consumed.json"
    output = tmp_path / "q-artifacts"
    p = _program(1)
    p_binding = {
        "formation_receipt_file_sha256": "1" * 64,
        "formation_receipt_hash": "2" * 64,
        "frozen_program_file_sha256": "3" * 64,
        "frozen_program_envelope_hash": "4" * 64,
        "program_hash": p.program_hash,
        "formed_on_block_id_hash": "5" * 64,
    }
    receipt = {"acquisition_sha256": "6" * 64}
    commitment = l4.BlockCommitment(
        block="F_Q",
        count=36,
        file_sha256="7" * 64,
        item_commitment_set_sha256="8" * 64,
    )
    implementation = {"files": [], "set_sha256": "9" * 64}
    private_open_count = 0

    monkeypatch.setattr(
        l4,
        "_load_acquisition",
        lambda **_kwargs: (receipt, b"acquisition\n"),
    )
    monkeypatch.setattr(l4, "_commitment", lambda *_args, **_kwargs: commitment)
    monkeypatch.setattr(
        l4,
        "_load_p",
        lambda **_kwargs: (p, p_binding),
    )
    monkeypatch.setattr(
        l4,
        "_assert_p_matches_preregistration",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        l4,
        "current_implementation_binding",
        lambda _project: implementation,
    )
    monkeypatch.setattr(
        l4,
        "_assert_git_ignored_private_path",
        lambda **_kwargs: marker,
    )

    def fail_after_marker(**_kwargs: Any):
        nonlocal private_open_count
        assert marker.is_file()
        private_open_count += 1
        raise l4.HotpotRecursiveL4Error("injected after marker")

    monkeypatch.setattr(l4, "_load_private_block", fail_after_marker)
    common = {
        "project_root": project,
        "acquisition_receipt_path": tmp_path / "acquisition.json",
        "f_q_block_path": tmp_path / "F_Q.jsonl",
        "p_formation_receipt_path": tmp_path / "p-formation.json",
        "p_frozen_program_path": tmp_path / "p-program.json",
        "output_dir": output,
    }
    with pytest.raises(l4.HotpotRecursiveL4Error, match="injected after marker"):
        l4.form_q(**common)
    assert private_open_count == 1
    marker_payload = json.loads(marker.read_text("utf-8"))
    marker_body = dict(marker_payload)
    marker_hash = marker_body.pop("consumption_sha256")
    assert marker_hash == stable_hash(marker_body)
    assert marker_body["formation_rows_opened_before_consumption"] == 0

    # Existence, rather than trusting marker content, is the replay barrier.
    marker.write_text("tampered but still terminal\n", encoding="utf-8")
    with pytest.raises(l4.HotpotRecursiveL4Error, match="already consumed"):
        l4.form_q(**common)
    assert private_open_count == 1
    assert not output.exists()


def test_load_q_rejects_marker_acquisition_and_p_cross_binding_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "q-artifacts"
    output.mkdir()
    formation_path = output / "formation.receipt.json"
    frozen_path = output / "frozen_program.json"
    marker_path = tmp_path / "q-formation.consumed.json"
    acquisition_raw = b"fixed acquisition receipt bytes\n"
    acquisition_receipt = {"acquisition_sha256": "a" * 64}
    commitment = l4.BlockCommitment(
        block="F_Q",
        count=36,
        file_sha256="b" * 64,
        item_commitment_set_sha256="c" * 64,
    )
    p = _program(1)
    q = _program(2)
    p_binding = {
        "formation_receipt_file_sha256": "1" * 64,
        "formation_receipt_hash": "2" * 64,
        "frozen_program_file_sha256": "3" * 64,
        "frozen_program_envelope_hash": "4" * 64,
        "program_hash": p.program_hash,
        "formed_on_block_id_hash": "5" * 64,
    }
    implementation = {"files": [], "set_sha256": "d" * 64}

    monkeypatch.setattr(
        l4,
        "_load_acquisition",
        lambda **_kwargs: (acquisition_receipt, acquisition_raw),
    )
    monkeypatch.setattr(l4, "_commitment", lambda *_args, **_kwargs: commitment)
    monkeypatch.setattr(l4, "_load_p", lambda **_kwargs: (p, p_binding))
    monkeypatch.setattr(
        l4,
        "_assert_p_matches_preregistration",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        l4,
        "current_implementation_binding",
        lambda _project: implementation,
    )
    monkeypatch.setattr(
        l4,
        "_assert_git_ignored_private_path",
        lambda **_kwargs: marker_path,
    )

    marker_body = {
        "schema": f"{l4.VERSION}_Q_formation_consumption",
        "acquisition_sha256": acquisition_receipt["acquisition_sha256"],
        "formation_block_id_hash": stable_hash({"block": "F_Q"}),
        "formation_block_file_sha256": commitment.file_sha256,
        "formation_item_commitment_set_sha256": (
            commitment.item_commitment_set_sha256
        ),
        "retained_P_program_hash": p.program_hash,
        "implementation_set_sha256": implementation["set_sha256"],
        "output_path_sha256": stable_hash(
            {"absolute_Q_formation_output": str(output)}
        ),
        "formation_rows_opened_before_consumption": 0,
        "retry_replay_resample_authorized": False,
    }

    def persist_marker(body: dict[str, Any]) -> bytes:
        payload = {**body, "consumption_sha256": stable_hash(body)}
        marker_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return marker_path.read_bytes()

    def persist_q(
        *,
        marker_raw: bytes,
        source_acquisition_sha256: str,
        formation_p_binding: dict[str, str],
    ) -> None:
        marker_payload = json.loads(marker_raw)
        formation_body = {
            "schema": l4.Q_FORMATION_SCHEMA,
            "status": "Q_formed_offline_on_exact_F_Q",
            "implementation": implementation,
            "source_binding": {
                "acquisition_file_sha256": l4._sha256_bytes(acquisition_raw),
                "acquisition_sha256": source_acquisition_sha256,
                "formation_block_id_hash": stable_hash({"block": "F_Q"}),
                "formation_block_file_sha256": commitment.file_sha256,
                "formation_item_commitment_set_sha256": (
                    commitment.item_commitment_set_sha256
                ),
                "formation_item_count": commitment.count,
            },
            "retained_P_binding": formation_p_binding,
            "prospective_ordering": {
                "formation_consumption_file_sha256": l4._sha256_bytes(marker_raw),
                "formation_consumption_sha256": marker_payload[
                    "consumption_sha256"
                ],
                "formation_rows_opened_before_consumption": 0,
                "formation_consumed_before_F_Q_open": True,
                "retry_replay_resample_authorized": False,
            },
            "selection": {"selected_program_hash": q.program_hash},
            "raw_content_persisted": False,
        }
        formation = {
            **formation_body,
            "formation_sha256": stable_hash(formation_body),
        }
        frozen_body = {
            "schema": l4.Q_PROGRAM_SCHEMA,
            "implementation": implementation,
            "program": q.to_dict(),
            "program_hash": q.program_hash,
            "formation_sha256": formation["formation_sha256"],
            "formation_block_id_hash": stable_hash({"block": "F_Q"}),
            "retained_P_program_hash": formation_p_binding["program_hash"],
            "raw_content_persisted": False,
        }
        frozen = {**frozen_body, "envelope_sha256": stable_hash(frozen_body)}
        formation_path.write_text(
            json.dumps(formation, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        frozen_path.write_text(
            json.dumps(frozen, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    marker_raw = persist_marker(marker_body)
    persist_q(
        marker_raw=marker_raw,
        source_acquisition_sha256=acquisition_receipt["acquisition_sha256"],
        formation_p_binding=p_binding,
    )
    loaded, _binding = l4.load_q(
        project=project,
        acquisition_receipt_path=tmp_path / "acquisition.json",
        p_formation_receipt_path=tmp_path / "p-formation.json",
        p_frozen_program_path=tmp_path / "p-program.json",
        formation_receipt_path=formation_path,
        frozen_program_path=frozen_path,
    )
    assert loaded.program_hash == q.program_hash

    tampered_marker = {**marker_body, "acquisition_sha256": "e" * 64}
    persist_marker(tampered_marker)
    with pytest.raises(l4.HotpotRecursiveL4Error, match="consumption drifted"):
        l4.load_q(
            project=project,
            acquisition_receipt_path=tmp_path / "acquisition.json",
            p_formation_receipt_path=tmp_path / "p-formation.json",
            p_frozen_program_path=tmp_path / "p-program.json",
            formation_receipt_path=formation_path,
            frozen_program_path=frozen_path,
        )

    marker_raw = persist_marker(marker_body)
    persist_q(
        marker_raw=marker_raw,
        source_acquisition_sha256="f" * 64,
        formation_p_binding=p_binding,
    )
    with pytest.raises(l4.HotpotRecursiveL4Error, match="formation/program drifted"):
        l4.load_q(
            project=project,
            acquisition_receipt_path=tmp_path / "acquisition.json",
            p_formation_receipt_path=tmp_path / "p-formation.json",
            p_frozen_program_path=tmp_path / "p-program.json",
            formation_receipt_path=formation_path,
            frozen_program_path=frozen_path,
        )

    wrong_p_binding = {**p_binding, "program_hash": "0" * 64}
    persist_q(
        marker_raw=marker_raw,
        source_acquisition_sha256=acquisition_receipt["acquisition_sha256"],
        formation_p_binding=wrong_p_binding,
    )
    with pytest.raises(l4.HotpotRecursiveL4Error, match="formation/program drifted"):
        l4.load_q(
            project=project,
            acquisition_receipt_path=tmp_path / "acquisition.json",
            p_formation_receipt_path=tmp_path / "p-formation.json",
            p_frozen_program_path=tmp_path / "p-program.json",
            formation_receipt_path=formation_path,
            frozen_program_path=frozen_path,
        )
