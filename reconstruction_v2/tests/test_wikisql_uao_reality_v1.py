from __future__ import annotations

import ast
from fractions import Fraction
import inspect
import itertools

import pytest

from assumption_agent.benchmarks import wikisql_uao_reality_v1 as core


SECRET = b"s" * core.HMAC_SECRET_BYTES
REVISION = core.canonical_sha256("synthetic-wikisql-release-1.1")


def _table() -> core.WikiSQLTable:
    return core.table_from_documented_schema(
        {
            "id": "synthetic-table",
            "header": ["City", "Population", "Note"],
            "types": ["text", "real", "text"],
            "rows": [
                ["Paris", "1,200", "North"],
                ["PARIS", 800, "South"],
                ["London", 1500.0, "North"],
                ["Berlin", 900, "West"],
                ["Rome", 950, "West"],
                ["Madrid", 975, "West"],
                ["Lisbon", 990, "West"],
                ["Oslo", 700, "West"],
                ["Bern", 650, "West"],
                ["Vienna", 600, "West"],
                ["Prague", 500, "West"],
            ],
        }
    )


def _hash(label: str) -> str:
    return core.canonical_sha256({"synthetic": label})


def _top5(*row_ids: int) -> tuple[int | None, ...]:
    if len(row_ids) > core.TOP_K:
        raise AssertionError("test top5 helper overflowed")
    return tuple(row_ids) + (None,) * (core.TOP_K - len(row_ids))


def test_core_has_no_source_filesystem_network_or_controller_import() -> None:
    tree = ast.parse(inspect.getsource(core))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports == {
        "__future__",
        "collections",
        "dataclasses",
        "fractions",
        "hashlib",
        "hmac",
        "json",
        "math",
        "re",
        "typing",
        "unicodedata",
    }
    assert not imports.intersection(
        {"pathlib", "os", "requests", "urllib", "sqlite3", "subprocess"}
    )


def test_canonical_json_and_hash_are_stable_and_fail_closed() -> None:
    left = {"β": [3, 2, 1], "a": {"z": True, "x": None}}
    right = {"a": {"x": None, "z": True}, "β": [3, 2, 1]}
    assert core.canonical_json_bytes(left) == core.canonical_json_bytes(right)
    assert core.canonical_sha256(left) == core.canonical_sha256(right)
    assert core.canonical_json_bytes(left).isascii()
    with pytest.raises(core.WikiSQLUAORealityError, match="canonical JSON"):
        core.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(core.WikiSQLUAORealityError, match="canonical JSON"):
        core.canonical_json_bytes({"bad": object()})


def test_source_native_single_where_family_partition_is_exact() -> None:
    expected = {
        0: "EQ",
        1: "GT",
        2: "LT",
    }
    for operator, family in expected.items():
        assert core.family_from_condition_operator(operator) == family
        assert core.family_from_condition_operator(family) == family
    for invalid in (-1, 3, True, "OP", "eq", "NONE", None):
        with pytest.raises(core.WikiSQLUAORealityError, match="EQ/GT/LT"):
            core.family_from_condition_operator(invalid)
    # Aggregation remains a validated SQL field but has no family authority.
    assert tuple(core.aggregation_name(index) for index in range(6)) == (
        "NONE",
        "MAX",
        "MIN",
        "COUNT",
        "SUM",
        "AVG",
    )


def test_item_identity_binds_revision_split_line_and_full_raw_object() -> None:
    raw_item = {
        "phase": 1,
        "question": "Which city has population above 1000?",
        "sql": {"sel": 0, "agg": 0, "conds": [[1, 1, 1000]]},
        "table_id": "synthetic-table",
    }
    identity = core.item_identity_commitment(
        source_revision_sha256=REVISION,
        split="test",
        line_number=7,
        raw_item=raw_item,
    )
    assert len(identity) == 64
    assert identity == core.item_identity_commitment(
        source_revision_sha256=REVISION,
        split="test",
        line_number=7,
        raw_item=dict(reversed(tuple(raw_item.items()))),
    )
    changed = dict(raw_item)
    changed["question"] = raw_item["question"] + " "
    assert identity != core.item_identity_commitment(
        source_revision_sha256=REVISION,
        split="test",
        line_number=7,
        raw_item=changed,
    )
    assert identity != core.item_identity_commitment(
        source_revision_sha256=REVISION,
        split="test",
        line_number=8,
        raw_item=raw_item,
    )
    with pytest.raises(core.WikiSQLUAORealityError, match="exact schema"):
        core.item_identity_commitment(
            source_revision_sha256=REVISION,
            split="test",
            line_number=7,
            raw_item={**raw_item, "answer": "leak"},
        )


def test_hmac_order_is_domain_separated_input_invariant_and_secret_private() -> None:
    candidates = tuple(
        core.SelectionCandidate(
            item_commitment_sha256=_hash(f"item-{index}"),
            table_commitment_sha256=_hash(f"table-{index}"),
            family=core.FAMILY_ORDER[index % 3],
            table_row_count=11,
            gold_row_count=1,
        )
        for index in range(9)
    )
    first = core.hmac_order(SECRET, block="A_hold", candidates=candidates)
    second = core.hmac_order(
        SECRET, block="A_hold", candidates=tuple(reversed(candidates))
    )
    assert first == second
    assert first != core.hmac_order(
        b"t" * core.HMAC_SECRET_BYTES,
        block="A_hold",
        candidates=candidates,
    )
    assert (
        core.hmac_selection_digest(
            SECRET,
            block="A_hold",
            family="EQ",
            item_commitment_sha256=candidates[0].item_commitment_sha256,
        )
        != core.hmac_selection_digest(
            SECRET,
            block="A_form",
            family="EQ",
            item_commitment_sha256=candidates[0].item_commitment_sha256,
        )
    )
    assert core.hmac_secret_commitment(SECRET) == _hash_bytes(SECRET)
    with pytest.raises(core.WikiSQLUAORealityError, match="32 bytes"):
        core.hmac_order(b"short", block="A_hold", candidates=candidates)


def _hash_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_split_local_hmac_cohort_matches_formal_block_selection() -> None:
    rows: list[core.SelectionCandidate] = []
    for family in core.FAMILY_ORDER:
        for index in range(89):
            rows.append(
                core.SelectionCandidate(
                    item_commitment_sha256=_hash(f"{family}-item-{index}"),
                    table_commitment_sha256=_hash(f"{family}-table-{index}"),
                    family=family,
                    table_row_count=11 + index % 70,
                    gold_row_count=1 + index % 5,
                )
            )
    # A second question from one already-represented table must not survive
    # table deduplication together with its sibling.
    rows.append(
        core.SelectionCandidate(
            item_commitment_sha256=_hash("EQ-duplicate-question"),
            table_commitment_sha256=_hash("EQ-table-0"),
            family="EQ",
            table_row_count=11,
            gold_row_count=1,
        )
    )
    first = core.select_hmac_cohort(
        SECRET,
        block="A_form",
        candidates=rows,
    )
    second = core.select_hmac_cohort(
        SECRET,
        block="A_form",
        candidates=tuple(reversed(rows)),
    )
    assert first == second
    assert len(first) == 192
    assert len({row.item_commitment_sha256 for row in first}) == 192
    assert len({row.table_commitment_sha256 for row in first}) == 192
    assert {
        family: sum(row.family == family for row in first)
        for family in core.FAMILY_ORDER
    } == core.COHORT_QUOTAS["A_form"]

    hold = core.select_hmac_cohort(
        SECRET,
        block="A_hold",
        candidates=rows,
    )
    assert len(hold) == 72
    assert {
        family: sum(row.family == family for row in hold)
        for family in core.FAMILY_ORDER
    } == core.COHORT_QUOTAS["A_hold"]
    # Cross-block disjointness is a source-compiler responsibility because
    # formal A_form and A_hold are selected from official TRAIN and TEST,
    # respectively.  This helper intentionally selects only one supplied
    # split-local pool.

    lt_keep = {_hash(f"LT-item-{index}") for index in range(63)}
    with pytest.raises(core.WikiSQLUAORealityError, match="quota"):
        core.select_hmac_cohort(
            SECRET,
            block="A_form",
            candidates=tuple(
                row
                for row in rows
                if row.family != "LT"
                or row.item_commitment_sha256 in lt_keep
            ),
        )
    with pytest.raises(core.WikiSQLUAORealityError, match="A_form or A_hold"):
        core.select_hmac_cohort(
            SECRET,
            block="M_search",
            candidates=rows,
        )


def test_selection_candidate_enforces_private_eligibility_counts() -> None:
    kwargs = {
        "item_commitment_sha256": _hash("candidate"),
        "table_commitment_sha256": _hash("candidate-table"),
        "family": "EQ",
        "table_row_count": 11,
        "gold_row_count": 1,
    }
    assert core.SelectionCandidate(**kwargs).family == "EQ"
    with pytest.raises(core.WikiSQLUAORealityError, match="11-through-80"):
        core.SelectionCandidate(**{**kwargs, "table_row_count": 10})
    with pytest.raises(core.WikiSQLUAORealityError, match="one-through-five"):
        core.SelectionCandidate(**{**kwargs, "gold_row_count": 6})


def test_documented_table_schema_and_row_serialization_are_exact() -> None:
    table = _table()
    assert table.table_id == "synthetic-table"
    assert table.header == ("City", "Population", "Note")
    first = core.serialize_table_row(table, 0)
    assert first == (
        'column[0] "City" (text) = Paris\n'
        'column[1] "Population" (real) = 1,200\n'
        'column[2] "Note" (text) = North'
    )
    assert core.serialize_table_rows(table)[2].endswith("(text) = North")
    assert "sql" not in first.casefold()
    assert "family" not in first.casefold()

    with pytest.raises(core.WikiSQLUAORealityError, match="exact schema"):
        core.table_from_documented_schema(
            {
                "id": "bad",
                "header": ["x"],
                "types": ["text"],
                "rows": [["x"]],
                "caption": "unfrozen field",
            }
        )
    with pytest.raises(core.WikiSQLUAORealityError, match="width"):
        core.WikiSQLTable("bad", ("a", "b"), ("text", "text"), (("x",),))
    with pytest.raises(core.WikiSQLUAORealityError, match="non-finite"):
        core.WikiSQLTable("bad", ("a",), ("real",), ((float("inf"),),))


def test_frozen_raw_bm25_ranks_rows_and_uses_ordinal_ties_with_null_padding() -> None:
    table = _table()
    result = core.raw_bm25_top5(
        "Which row is London with population 1500?", table
    )
    assert result[0] == 2
    assert len(result) == core.TOP_K
    assert None not in result

    # No row contains the token; all zero-score ties resolve by row ordinal.
    assert core.raw_bm25_top5("xyzzynotpresent", table) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert core.bm25_scores(
        "London 1500", core.serialize_table_rows(table)
    ) == core.bm25_scores(
        "London 1500", core.serialize_table_rows(table)
    )

    short = core.WikiSQLTable(
        "short",
        ("value",),
        ("text",),
        tuple((str(index),) for index in range(10)),
    )
    with pytest.raises(core.WikiSQLUAORealityError, match="11-through-80"):
        core.raw_bm25_top5("value", short)


def test_gold_rows_follow_single_where_lowercase_and_numeric_semantics() -> None:
    table = _table()
    eq_sql = {
        "sel": 1,
        "agg": 1,
        "conds": [[0, 0, "paris"]],
    }
    query = core.query_from_documented_sql(eq_sql, column_count=3)
    assert query.family == "EQ"
    assert core.derive_gold_row_ids(table, query) == (0, 1)
    assert core.derive_eligible_gold_label(table, query) == core.EligibleGoldLabel(
        family="EQ",
        table_row_count=11,
        gold_row_ids=(0, 1),
    )

    # Aggregation is deliberately irrelevant: family follows the one WHERE
    # relation and every matching physical row remains gold.
    assert core.derive_gold_row_ids(
        table,
        {"sel": 0, "agg": 5, "conds": [[1, 1, "1,000"]]},
    ) == (0, 2)
    assert core.derive_eligible_gold_label(
        table,
        {"sel": 0, "agg": 5, "conds": [[1, 1, "1,000"]]},
    ).family == "GT"
    assert core.derive_gold_row_ids(
        table,
        {"sel": 0, "agg": 0, "conds": [[2, 0, "NORTH"]]},
    ) == (0, 2)
    assert core.derive_gold_row_ids(
        table,
        {"sel": 0, "agg": 3, "conds": [[1, 2, "700"]]},
    ) == (8, 9, 10)


def test_numeric_fallback_matches_official_first_regex_match() -> None:
    table = core.WikiSQLTable(
        "numeric-fallback",
        ("value",),
        ("real",),
        (
            ("-12.5 kg",),
            ("12 kg x 34",),
            (34,),
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
        ),
    )
    assert core.derive_gold_row_ids(
        table,
        {"sel": 0, "agg": 0, "conds": [[0, 0, "-12.5 units"]]},
    ) == (0,)
    assert core.derive_gold_row_ids(
        table,
        {"sel": 0, "agg": 0, "conds": [[0, 0, "abc 12 x 34"]]},
    ) == (1,)


def test_gold_derivation_rejects_schema_operator_column_and_coercion_drift() -> None:
    table = _table()
    with pytest.raises(core.WikiSQLUAORealityError, match="exact schema"):
        core.derive_gold_row_ids(
            table,
            {"sel": 0, "agg": 0, "conds": [], "answer": "leak"},
        )
    with pytest.raises(core.WikiSQLUAORealityError, match="operator"):
        core.derive_gold_row_ids(
            table,
            {"sel": 0, "agg": 0, "conds": [[0, 3, "Paris"]]},
        )
    for conditions in ([], [[0, 0, "Paris"], [2, 0, "North"]]):
        with pytest.raises(core.WikiSQLUAORealityError, match="exactly one"):
            core.derive_gold_row_ids(
                table,
                {"sel": 0, "agg": 0, "conds": conditions},
            )
    with pytest.raises(core.WikiSQLUAORealityError, match="outside"):
        core.derive_gold_row_ids(
            table,
            {"sel": 3, "agg": 0, "conds": [[0, 0, "Paris"]]},
        )
    malformed = core.WikiSQLTable(
        "bad-real",
        ("value",),
        ("real",),
        (("not-a-number",),),
    )
    with pytest.raises(core.WikiSQLUAORealityError, match="coerced"):
        core.derive_gold_row_ids(
            malformed,
            {"sel": 0, "agg": 0, "conds": [[0, 0, 1]]},
        )


def test_eligible_label_enforces_table_and_gold_cardinality() -> None:
    table = _table()
    assert core.derive_eligible_gold_label(
        table,
        {"sel": 0, "agg": 0, "conds": [[0, 0, "Paris"]]},
    ).gold_row_ids == (0, 1)

    short = core.WikiSQLTable(
        "short",
        ("value",),
        ("real",),
        tuple((index,) for index in range(10)),
    )
    with pytest.raises(core.WikiSQLUAORealityError, match="11 through 80"):
        core.derive_eligible_gold_label(
            short,
            {"sel": 0, "agg": 0, "conds": [[0, 1, 5]]},
        )

    too_many = core.derive_gold_row_ids(
        table,
        {"sel": 0, "agg": 0, "conds": [[1, 1, 0]]},
    )
    assert len(too_many) == 11
    with pytest.raises(core.WikiSQLUAORealityError, match="one through five"):
        core.derive_eligible_gold_label(
            table,
            {"sel": 0, "agg": 0, "conds": [[1, 1, 0]]},
        )
    with pytest.raises(core.WikiSQLUAORealityError, match="one through five"):
        core.derive_eligible_gold_label(
            table,
            {"sel": 0, "agg": 0, "conds": [[0, 0, "Tokyo"]]},
        )


def test_integer_hits_plus_complete_utility_and_top5_validation() -> None:
    incomplete = core.score_item(_top5(0, 1), (0, 2))
    assert incomplete == core.ItemUtility(hits=1, complete=False, utility=1)
    complete = core.score_item(_top5(0, 2), (0, 2))
    assert complete == core.ItemUtility(hits=2, complete=True, utility=3)
    assert core.item_utility(_top5(3), (3,)) == 2

    with pytest.raises(core.WikiSQLUAORealityError, match="repeats"):
        core.item_utility((0, 0, None, None, None), (0,))
    with pytest.raises(core.WikiSQLUAORealityError, match="trailing"):
        core.item_utility((0, None, 1, None, None), (0,))
    with pytest.raises(core.WikiSQLUAORealityError, match="one-through-five"):
        core.item_utility(_top5(0), ())
    with pytest.raises(core.WikiSQLUAORealityError, match="one-through-five"):
        core.item_utility(_top5(0), (0, 1, 2, 3, 4, 5))


def _brute_sign_flip(deltas: tuple[int, ...]) -> Fraction:
    magnitudes = tuple(abs(value) for value in deltas if value)
    observed = sum(deltas)
    values = tuple(
        sum(sign * value for sign, value in zip(signs, magnitudes, strict=True))
        for signs in itertools.product((-1, 1), repeat=len(magnitudes))
    )
    return Fraction(sum(value >= observed for value in values), len(values))


@pytest.mark.parametrize(
    "deltas",
    (
        (1, 1, 1, 1),
        (2, 1, -1, 0),
        (3, -2, 1, 0, 0),
        (-1, -1, -1),
        (0, 0, 0),
    ),
)
def test_exact_sign_flip_matches_bruteforce_and_keeps_ties(deltas: tuple[int, ...]) -> None:
    result = core.exact_magnitude_preserving_sign_flip(deltas)
    assert result.observed_net_u == sum(deltas)
    assert result.nonzero_pair_count == sum(value != 0 for value in deltas)
    assert result.p_value == _brute_sign_flip(deltas)
    if deltas == (1, 1, 1, 1):
        assert result.p_value == Fraction(1, 16)
        assert result.positive_at_alpha is True
    if deltas == (0, 0, 0):
        assert result.p_value == 1
        assert result.positive_at_alpha is False


def _passing_measurements() -> tuple[core.ItemMeasurement, ...]:
    rows: list[core.ItemMeasurement] = []
    for family in core.FAMILY_ORDER:
        for index in range(2):
            rows.append(
                core.ItemMeasurement(
                    item_commitment_sha256=_hash(f"score-{family}-{index}"),
                    family=family,
                    gold_row_ids=(0,),
                    agent_top5=_top5(0, 1, 2, 3, 4),
                    raw_top5=_top5(1, 2, 3, 4, 5),
                    hipporag_top5=_top5(1, 2, 3, 4, 5),
                )
            )
    return tuple(rows)


def test_primary_requires_aggregate_p_tail_and_all_three_family_nets_vs_both() -> None:
    result = core.aggregate_primary(_passing_measurements())
    assert result.item_count == 6
    assert result.family_counts == (
        ("EQ", 2),
        ("GT", 2),
        ("LT", 2),
    )
    assert result.agent_vs_raw.observed_net_u == 12
    assert result.agent_vs_raw.family_net_u == (
        ("EQ", 4),
        ("GT", 4),
        ("LT", 4),
    )
    assert result.agent_vs_raw.sign_flip.p_value == Fraction(1, 64)
    assert result.agent_vs_raw.passed is True
    assert result.agent_vs_hipporag.passed is True
    assert result.passed is True

    rows = list(_passing_measurements())
    # Keep enough aggregate positive pairs for p<=.10, but force the EQ
    # family net against RAW to zero.  Aggregate significance cannot rescue it.
    rows[0] = core.ItemMeasurement(
        item_commitment_sha256=rows[0].item_commitment_sha256,
        family=rows[0].family,
        gold_row_ids=rows[0].gold_row_ids,
        agent_top5=rows[0].agent_top5,
        raw_top5=rows[0].agent_top5,
        hipporag_top5=rows[0].hipporag_top5,
    )
    rows[1] = core.ItemMeasurement(
        item_commitment_sha256=rows[1].item_commitment_sha256,
        family=rows[1].family,
        gold_row_ids=rows[1].gold_row_ids,
        agent_top5=rows[1].agent_top5,
        raw_top5=rows[1].agent_top5,
        hipporag_top5=rows[1].hipporag_top5,
    )
    failed = core.aggregate_primary(rows)
    assert failed.agent_vs_raw.observed_net_u == 8
    assert failed.agent_vs_raw.sign_flip.p_value == Fraction(1, 16)
    assert failed.agent_vs_raw.family_net_u[0] == ("EQ", 0)
    assert failed.agent_vs_raw.passed is False
    assert failed.agent_vs_hipporag.passed is True
    assert failed.passed is False


def test_primary_fails_closed_on_missing_family_or_duplicate_item() -> None:
    rows = _passing_measurements()
    with pytest.raises(core.WikiSQLUAORealityError, match="all three"):
        core.aggregate_primary(
            tuple(row for row in rows if row.family != "LT")
        )
    with pytest.raises(core.WikiSQLUAORealityError, match="repeat"):
        core.aggregate_primary(rows + (rows[0],))
