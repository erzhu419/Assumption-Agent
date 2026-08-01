from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from hegel_machine.phase3_m25_split_v1 import (
    DISCOVERY_PARTITION_ID,
    NULL_CONTROL_ROLE_ID,
    ODD_STRATUM_QUOTAS,
    OUTSIDE_ROLE_ID,
    SEALED_PREDICTION_PARTITION_ID,
    SINK_STRATUM_QUOTAS,
    VALIDATION_PARTITION_ID,
    SplitRankInput,
    allocate_split_rows,
    allocate_typed_role_rows,
    assert_authoritative_seed_genesis_available,
    derive_role_key,
    split_partition_commitments,
    typed_role_split_rank_inputs,
)
from hegel_machine.phase3_m25_rows_v1 import (
    M25TypedRowError,
    TypedRoleRows,
    generate_odd_role_rows_v1,
    generate_sink_role_rows_v1,
)
from hegel_machine.phase3_m25_wire_v1 import M25WireError


def _rows_for_quotas(quotas: object) -> tuple[SplitRankInput, ...]:
    rows: list[SplitRankInput] = []
    assert hasattr(quotas, "items")
    for stratum_id, quota in quotas.items():
        for _ in range(quota.universe):
            index = len(rows)
            rows.append(
                SplitRankInput(
                    canonical_input_hash=hashlib.sha256(
                        f"synthetic-row:{index}".encode("ascii")
                    ).digest(),
                    row_identity=f"row:{index}".encode("ascii"),
                    stratum_id=stratum_id,
                    universe_index=index,
                )
            )
    return tuple(rows)


def _partition_counts(assignments: object) -> dict[int, int]:
    result = {
        DISCOVERY_PARTITION_ID: 0,
        VALIDATION_PARTITION_ID: 0,
        SEALED_PREDICTION_PARTITION_ID: 0,
    }
    for row in assignments:
        result[row.partition_id] += 1
    return result


def test_frozen_quota_tables_are_exhaustive_and_have_exact_totals() -> None:
    assert tuple(ODD_STRATUM_QUOTAS) == tuple(range(1, 9))
    assert tuple(SINK_STRATUM_QUOTAS) == tuple(range(9, 14))
    assert sum(row.universe for row in ODD_STRATUM_QUOTAS.values()) == 480
    assert sum(row.discovery for row in ODD_STRATUM_QUOTAS.values()) == 192
    assert sum(row.validation for row in ODD_STRATUM_QUOTAS.values()) == 96
    assert sum(row.sealed for row in ODD_STRATUM_QUOTAS.values()) == 192
    assert sum(row.universe for row in SINK_STRATUM_QUOTAS.values()) == 85
    assert sum(row.discovery for row in SINK_STRATUM_QUOTAS.values()) == 39
    assert sum(row.validation for row in SINK_STRATUM_QUOTAS.values()) == 20
    assert sum(row.sealed for row in SINK_STRATUM_QUOTAS.values()) == 26


def test_odd_allocation_ranks_within_each_stratum_and_commits_partitions() -> None:
    rows = _rows_for_quotas(ODD_STRATUM_QUOTAS)
    role_key = derive_role_key(bytes(range(32)), OUTSIDE_ROLE_ID)
    assignments = allocate_split_rows(
        role_key,
        OUTSIDE_ROLE_ID,
        rows,
        ODD_STRATUM_QUOTAS,
    )
    assert len(assignments) == 480
    assert tuple(row.universe_index for row in assignments) == tuple(range(480))
    assert _partition_counts(assignments) == {1: 192, 2: 96, 3: 192}
    for stratum_id, quota in ODD_STRATUM_QUOTAS.items():
        selected = [row for row in assignments if row.stratum_id == stratum_id]
        assert _partition_counts(selected) == {
            1: quota.discovery,
            2: quota.validation,
            3: quota.sealed,
        }
    commitments = split_partition_commitments(OUTSIDE_ROLE_ID, assignments)
    assert (
        commitments.discovery_count,
        commitments.validation_count,
        commitments.sealed_count,
    ) == (192, 96, 192)
    assert len(commitments.discovery_root) == 32
    assert len(commitments.validation_root) == 32
    assert len(commitments.sealed_root) == 32


def test_sink_allocation_uses_d_strata_and_exact_39_20_26_quota() -> None:
    rows = _rows_for_quotas(SINK_STRATUM_QUOTAS)
    role_key = derive_role_key(bytes(range(32)), NULL_CONTROL_ROLE_ID)
    assignments = allocate_split_rows(
        role_key,
        NULL_CONTROL_ROLE_ID,
        rows,
        SINK_STRATUM_QUOTAS,
    )
    assert len(assignments) == 85
    assert _partition_counts(assignments) == {1: 39, 2: 20, 3: 26}
    commitments = split_partition_commitments(NULL_CONTROL_ROLE_ID, assignments)
    assert (
        commitments.discovery_count,
        commitments.validation_count,
        commitments.sealed_count,
    ) == (39, 20, 26)


@pytest.mark.parametrize("role_name", ["odd", "sink"])
def test_typed_allocator_derives_index_hash_and_semantic_stratum(
    role_name: str,
) -> None:
    rows = (
        generate_odd_role_rows_v1()
        if role_name == "odd"
        else generate_sink_role_rows_v1()
    )
    role_id, rank_inputs, quotas = typed_role_split_rank_inputs(rows)
    assignments = allocate_typed_role_rows(
        derive_role_key(bytes(range(32)), role_id),
        rows,
    )
    assert len(rank_inputs) == len(assignments) == len(rows.universe_rows)
    assert tuple(item.universe_index for item in rank_inputs) == tuple(
        range(len(rank_inputs))
    )
    assert tuple(item.canonical_input_hash for item in rank_inputs) == tuple(
        row[4] for row in rows.truth_rows
    )
    for rank_input, universe_row, truth_row in zip(
        rank_inputs,
        rows.universe_rows,
        rows.truth_rows,
        strict=True,
    ):
        expected_stratum = (
            1 + 2 * (universe_row[5][3] - 5) + truth_row[5]
            if role_name == "odd"
            else 9 + universe_row[5][6]
        )
        assert rank_input.stratum_id == expected_stratum
        assert expected_stratum in quotas
    assert _partition_counts(assignments) == (
        {1: 192, 2: 96, 3: 192}
        if role_name == "odd"
        else {1: 39, 2: 20, 3: 26}
    )


def test_typed_allocator_rejects_role_name_signature_mismatch() -> None:
    odd = generate_odd_role_rows_v1()
    misnamed = TypedRoleRows(
        role_name="sink",
        input_signature_id=odd.input_signature_id,
        universe_rows=odd.universe_rows,
        truth_rows=odd.truth_rows,
    )
    with pytest.raises(M25WireError) as error:
        typed_role_split_rank_inputs(misnamed)
    assert error.value.code == "FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH"


def test_typed_allocator_rejects_truth_hash_detached_from_universe_row() -> None:
    sink = generate_sink_role_rows_v1()
    altered_truth = list(sink.truth_rows)
    altered_truth[0] = (*altered_truth[0][:4], bytes(32), altered_truth[0][5])
    detached = TypedRoleRows(
        role_name=sink.role_name,
        input_signature_id=sink.input_signature_id,
        universe_rows=sink.universe_rows,
        truth_rows=tuple(altered_truth),
    )
    with pytest.raises(M25TypedRowError) as error:
        typed_role_split_rank_inputs(detached)
    assert error.value.code == "FAIL_CANONICAL_INPUT_HASH_MISMATCH"


@pytest.mark.parametrize("mutation", ["duplicate", "gap", "quota", "stratum"])
def test_allocation_rejects_nonexhaustive_or_misbound_rows(mutation: str) -> None:
    rows = list(_rows_for_quotas(SINK_STRATUM_QUOTAS))
    if mutation == "duplicate":
        rows[-1] = SplitRankInput(
            rows[-1].canonical_input_hash,
            rows[-1].row_identity,
            rows[-1].stratum_id,
            rows[-2].universe_index,
        )
        expected = "FAIL_SPLIT_UNIVERSE_INDEX_DUPLICATE"
    elif mutation == "gap":
        rows[-1] = SplitRankInput(
            rows[-1].canonical_input_hash,
            rows[-1].row_identity,
            rows[-1].stratum_id,
            100,
        )
        expected = "FAIL_SPLIT_UNIVERSE_INDEX_GAP"
    elif mutation == "quota":
        rows.pop()
        expected = "FAIL_SPLIT_QUOTA_MISMATCH"
    else:
        last = rows[-1]
        rows[-1] = SplitRankInput(
            last.canonical_input_hash,
            last.row_identity,
            99,
            last.universe_index,
        )
        expected = "FAIL_SPLIT_QUOTA_MISMATCH"
    with pytest.raises(M25WireError) as error:
        allocate_split_rows(
            derive_role_key(bytes(32), NULL_CONTROL_ROLE_ID),
            NULL_CONTROL_ROLE_ID,
            rows,
            SINK_STRATUM_QUOTAS,
        )
    assert error.value.code == expected


def test_local_process_cannot_claim_independent_custodian_authority() -> None:
    with pytest.raises(M25WireError) as error:
        assert_authoritative_seed_genesis_available()
    assert error.value.code == "FAIL_CUSTODIAN_KEY_MISSING"


def test_role_cannot_use_the_other_frozen_quota_registry() -> None:
    with pytest.raises(M25WireError) as error:
        allocate_split_rows(
            derive_role_key(bytes(32), OUTSIDE_ROLE_ID),
            OUTSIDE_ROLE_ID,
            _rows_for_quotas(SINK_STRATUM_QUOTAS),
            SINK_STRATUM_QUOTAS,
        )
    assert error.value.code == "FAIL_SPLIT_QUOTA_MISMATCH"


@pytest.mark.parametrize("mutation", ["partition", "duplicate", "hash"])
def test_partition_commitment_rejects_forged_assignment_sets(mutation: str) -> None:
    rows = _rows_for_quotas(SINK_STRATUM_QUOTAS)
    assignments = list(
        allocate_split_rows(
            derive_role_key(bytes(32), NULL_CONTROL_ROLE_ID),
            NULL_CONTROL_ROLE_ID,
            rows,
            SINK_STRATUM_QUOTAS,
        )
    )
    if mutation == "partition":
        assignments[0] = replace(assignments[0], partition_id=99)
        expected = "FAIL_SPLIT_QUOTA_MISMATCH"
    elif mutation == "duplicate":
        assignments[-1] = replace(
            assignments[-1],
            universe_index=assignments[-2].universe_index,
        )
        expected = "FAIL_SPLIT_UNIVERSE_INDEX_DUPLICATE"
    else:
        assignments[0] = replace(assignments[0], rank_digest=b"short")
        expected = "REJECT_M25_CRYPTO_INPUT"
    with pytest.raises(M25WireError) as error:
        split_partition_commitments(NULL_CONTROL_ROLE_ID, assignments)
    assert error.value.code == expected
