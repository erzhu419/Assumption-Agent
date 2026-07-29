from __future__ import annotations

import copy
import hashlib

import pytest

from assumption_agent.benchmarks import wikisql_uao_scorer_v1 as scorer


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hashed(base: dict[str, object]) -> dict[str, object]:
    return {**base, "self_sha256": scorer.canonical_sha256(base)}


def _cohort() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    identifiers = tuple(sorted(_sha(f"opaque-{index}") for index in range(72)))
    views: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    families = ("EQ",) * 24 + ("GT",) * 24 + ("LT",) * 24
    # Family allocation follows sorted opaque IDs, exactly as formal packs do.
    for index, (item_id, family) in enumerate(zip(identifiers, families, strict=True)):
        view = {
            "opaque_item_id": item_id,
            "physical_rows": [
                [f"name-{index}-{row}", row] for row in range(11)
            ],
            "question": f"Which Name has Score equal to 0 for item {index}?",
            "table_header": ["Name", "Score"],
            "table_types": ["text", "real"],
        }
        views.append(view)
        labels.append(
            {
                "action_view_sha256": scorer.canonical_sha256(view),
                "family": family,
                "gold_row_ids": [0],
                "item_commitment_sha256": _sha(f"source-{index}"),
                "opaque_item_id": item_id,
                "sqlite_rowid_cross_checked": True,
                "table_row_count": 11,
            }
        )
    view_pack = _hashed(
        {
            "block": "A_hold",
            "contains_labels": False,
            "item_count": 72,
            "items": views,
            "schema": scorer.ACTION_VIEW_PACK_SCHEMA,
            "study_id": scorer.STUDY_ID,
        }
    )
    label_pack = scorer.build_minimal_label_pack(
        action_view_pack_sha256=view_pack["self_sha256"],
        items=labels,
    )

    def actions(arm: str, top5: list[int]) -> dict[str, object]:
        return _hashed(
            {
                "action_view_pack_sha256": view_pack["self_sha256"],
                "arm": arm,
                "block": "A_hold",
                "item_count": 72,
                "items": [
                    {
                        "opaque_item_id": item_id,
                        "top5_row_ids": list(top5),
                    }
                    for item_id in identifiers
                ],
                "schema": scorer.ACTION_PACK_SCHEMA,
                "study_id": scorer.STUDY_ID,
            }
        )

    return (
        view_pack,
        label_pack,
        actions("Agent", [0, 1, 2, 3, 4]),
        actions("RAW", [5, 6, 7, 8, 9]),
        actions("HippoRAG", [5, 6, 7, 8, 9]),
    )


def _rehash(pack: dict[str, object]) -> dict[str, object]:
    base = {key: value for key, value in pack.items() if key != "self_sha256"}
    return _hashed(base)


def _score(
    packs: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
) -> scorer.ScoringArtifacts:
    view, labels, agent, raw, hippo = packs
    return scorer.score_late_labels(
        action_view_pack=view,
        minimal_label_pack=labels,
        agent_action_pack=agent,
        raw_action_pack=raw,
        hipporag_action_pack=hippo,
    )


def test_late_label_pass_outputs_private_items_and_safe_aggregate_only() -> None:
    artifacts = _score(_cohort())
    private = artifacts.private_score_pack
    safe = artifacts.safe_aggregate_receipt

    assert private["schema"] == scorer.PRIVATE_SCORE_PACK_SCHEMA
    assert private["item_count"] == 72
    assert len(private["items"]) == 72
    assert private["self_sha256"] == scorer.canonical_sha256(
        {key: value for key, value in private.items() if key != "self_sha256"}
    )
    assert private["items"][0]["Agent"]["utility"] == 2
    assert private["items"][0]["RAW"]["utility"] == 0

    assert safe["schema"] == scorer.SAFE_AGGREGATE_SCHEMA
    assert safe["status"] == "PASS_REALITY_PRIMARY"
    assert safe["primary_passed"] is True
    assert safe["family_counts"] == {"EQ": 24, "GT": 24, "LT": 24}
    for comparison in ("Agent_vs_RAW", "Agent_vs_HippoRAG"):
        result = safe[comparison]
        assert result["observed_net_u"] == 144
        assert result["family_net_u"] == {"EQ": 48, "GT": 48, "LT": 48}
        assert result["exact_p_numerator"] == 1
        assert result["exact_p_denominator"] == 1 << 72
        assert result["passed"] is True
    assert safe["private_score_pack_sha256"] == private["self_sha256"]
    assert safe["self_sha256"] == scorer.canonical_sha256(
        {key: value for key, value in safe.items() if key != "self_sha256"}
    )
    safe_text = scorer.canonical_json_bytes(safe).decode("ascii")
    for forbidden in (
        "opaque_item_id",
        "gold_row_ids",
        "top5_row_ids",
        '"items"',
        '"utility"',
    ):
        assert forbidden not in safe_text


def test_unbalanced_late_labels_fail_closed() -> None:
    packs = list(_cohort())
    labels = copy.deepcopy(packs[1])
    labels["items"][0]["family"] = "GT"
    packs[1] = _rehash(labels)
    with pytest.raises(scorer.WikiSQLUAOScorerError, match="balanced 24x"):
        _score(tuple(packs))


def test_missing_action_id_set_fails_closed() -> None:
    packs = list(_cohort())
    agent = copy.deepcopy(packs[2])
    alien = _sha("alien-action-id")
    assert alien not in {
        row["opaque_item_id"] for row in agent["items"]
    }
    agent["items"][0]["opaque_item_id"] = alien
    agent["items"].sort(key=lambda row: row["opaque_item_id"])
    packs[2] = _rehash(agent)
    with pytest.raises(scorer.WikiSQLUAOScorerError, match="not identical"):
        _score(tuple(packs))


@pytest.mark.parametrize(
    ("invalid_top5", "message"),
    (
        ([5, 6, 7, 8, 11], "exceeds table row count"),
        ([5, 6, 7, 8, None], "five distinct integer ordinals"),
        ([5, 6, 7, 8, 8], "five distinct integer ordinals"),
    ),
)
def test_out_of_range_padding_and_duplicate_actions_fail_closed(
    invalid_top5: list[int | None],
    message: str,
) -> None:
    packs = list(_cohort())
    hippo = copy.deepcopy(packs[4])
    hippo["items"][0]["top5_row_ids"] = invalid_top5
    packs[4] = _rehash(hippo)
    with pytest.raises(scorer.WikiSQLUAOScorerError, match=message):
        _score(tuple(packs))


def test_pack_hash_tamper_and_view_label_binding_fail_closed() -> None:
    tampered = list(_cohort())
    tampered[3]["items"][0]["top5_row_ids"] = [0, 1, 2, 3, 4]
    with pytest.raises(scorer.WikiSQLUAOScorerError, match="content hash"):
        _score(tuple(tampered))

    rebound = list(_cohort())
    labels = copy.deepcopy(rebound[1])
    labels["items"][0]["action_view_sha256"] = "f" * 64
    rebound[1] = _rehash(labels)
    with pytest.raises(
        scorer.WikiSQLUAOScorerError,
        match="action_view_sha256.*binding",
    ):
        _score(tuple(rebound))


def test_all_common_action_packs_bind_exact_same_action_view_pack() -> None:
    packs = list(_cohort())
    raw = copy.deepcopy(packs[3])
    raw["action_view_pack_sha256"] = "a" * 64
    packs[3] = _rehash(raw)
    with pytest.raises(scorer.WikiSQLUAOScorerError, match="envelope"):
        _score(tuple(packs))
