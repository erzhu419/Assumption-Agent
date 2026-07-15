from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks.historical_typed_selection_projection_v2 import (
    HISTORICAL_PROJECTED_LEDGER_HASH,
    HistoricalTypedSelectionProjectionError,
    load_historical_portable_typed_selection_projection_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01"
)


@pytest.mark.skipif(
    not SOURCE_RUN_ROOT.is_dir(),
    reason="historical offline TRAIN source is not installed",
)
def test_reconstructs_exact_historical_projected_ledger_without_authority() -> None:
    receipt = load_historical_portable_typed_selection_projection_v2(
        project_root=PROJECT_ROOT,
    )

    receipt.verify()
    assert receipt.projected_snapshot_ledger_hash == (
        HISTORICAL_PROJECTED_LEDGER_HASH
    )
    assert receipt.ledger.production_snapshot_ledger.ledger_hash == (
        HISTORICAL_PROJECTED_LEDGER_HASH
    )
    assert receipt.safe_payload()["snapshot_count"] == 3
    assert receipt.safe_payload()["train_observation_count"] == 38
    assert receipt.safe_payload()["model_calls"] == 0
    assert receipt.safe_payload()["evaluator_calls"] == 0
    assert receipt.ledger.freeze_authorization is None
    with pytest.raises(PermissionError, match="diagnostic-only"):
        receipt.ledger.require_freeze_authorization()

    with pytest.raises(
        HistoricalTypedSelectionProjectionError,
        match="source receipt drifted",
    ):
        replace(
            receipt,
            projected_snapshot_ledger_hash="0" * 64,
        ).verify()
