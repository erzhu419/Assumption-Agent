from pathlib import Path

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_direct_c_confirm_v1 as runtime,
)


def _items() -> tuple[runtime.RuntimeItem, ...]:
    rows = []
    ordinal = 0
    for family in runtime.FAMILIES:
        for attempt in range(runtime.ATTEMPTS_PER_FAMILY):
            rows.append(
                runtime.RuntimeItem(
                    ordinal=ordinal,
                    family=family,
                    attempt_ordinal=attempt,
                    family_hmac_position=attempt,
                    item_key=f"{family}-{attempt}",
                    query="query",
                    source_query_id=f"q-{attempt}",
                    excluded_ids=(),
                )
            )
            ordinal += 1
    return tuple(rows)


def test_complete_case_selection_uses_first_terminal_hmac_positions() -> None:
    items = _items()
    terminal = tuple(
        item.ordinal
        for item in items
        if item.attempt_ordinal not in {0, 3, 7}
    )
    passed, selected, counts = runtime.select_complete_cases(items, terminal)
    assert passed is True
    assert counts == {family: 17 for family in runtime.FAMILIES}
    assert {
        family: [
            item.attempt_ordinal for item in selected if item.family == family
        ]
        for family in runtime.FAMILIES
    } == {
        family: [1, 2, 4, 5, 6, 8, 9, 10, 11, 12]
        for family in runtime.FAMILIES
    }


def test_capacity_failure_selects_no_partial_complete_case() -> None:
    items = _items()
    terminal = tuple(
        item.ordinal
        for item in items
        if item.family != runtime.FAMILIES[0]
        or item.attempt_ordinal < 9
    )
    passed, selected, counts = runtime.select_complete_cases(items, terminal)
    assert passed is False
    assert selected == ()
    assert counts[runtime.FAMILIES[0]] == 9


def test_primary_requires_positive_net_for_every_family_and_baseline() -> None:
    items = tuple(
        item
        for item in _items()
        if item.attempt_ordinal < runtime.TARGET_PER_FAMILY
    )
    scores = {
        "Agent": [3] * runtime.SELECTED_COUNT,
        "RAW": [2] * runtime.SELECTED_COUNT,
        "HippoRAG": [1] * runtime.SELECTED_COUNT,
    }
    passed, comparisons = runtime.primary_decision(
        items=items, arm_scores=scores
    )
    assert passed is True
    assert comparisons["Agent_minus_RAW"]["net_integer_ndcg"] == 30
    harmed = {name: list(values) for name, values in scores.items()}
    harmed["HippoRAG"][0:10] = [4] * 10
    passed, _comparisons = runtime.primary_decision(
        items=items, arm_scores=harmed
    )
    assert passed is False


def test_terminal_receipt_requires_graph_and_unique_top10() -> None:
    receipt = {
        "graph_edge_count": 1,
        "graph_node_count": 33,
        "top_rows": list(range(10)),
    }
    assert runtime._valid_terminal_receipt(receipt, tuple(range(32))) is True
    receipt["top_rows"] = [0] * 10
    assert runtime._valid_terminal_receipt(receipt, tuple(range(32))) is False


def test_formal_refuses_consumed_root_before_private_access(
    tmp_path: Path,
) -> None:
    base = tmp_path / "reconstruction_v2"
    (base / runtime.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(runtime.OneShotRefusal, match="root already exists"):
        runtime.run_formal(tmp_path)
