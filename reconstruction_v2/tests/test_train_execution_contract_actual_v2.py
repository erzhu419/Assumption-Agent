from __future__ import annotations

import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks.train_execution_contract_actual_v2 import (
    MODEL_INFERENCE_SLOTS,
    OUTER_WORKERS,
    TrainExecutionContractActualError,
    _verify_canary,
)


def _write_canary(path: Path, *, accepted: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "canary_version": "proposal_canary_v1",
                "model": "gpt-5.4-mini",
                "provider_chain": ["openai_compatible"],
                "accepted": accepted,
                "api_key_present": True,
                "secret_value_persisted": False,
                "raw_content_persisted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_provider_canary_authorizes_only_a_completed_fixed_model_probe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canary.json"
    _write_canary(path, accepted=True)

    receipt = _verify_canary(path, provider_label="plus")
    assert receipt["canary_accepted"] is True
    assert receipt["secret_value_persisted"] is False
    assert OUTER_WORKERS == 56
    assert MODEL_INFERENCE_SLOTS == 48

    _write_canary(path, accepted=False)
    with pytest.raises(
        TrainExecutionContractActualError,
        match="did not authorize",
    ):
        _verify_canary(path, provider_label="plus")
