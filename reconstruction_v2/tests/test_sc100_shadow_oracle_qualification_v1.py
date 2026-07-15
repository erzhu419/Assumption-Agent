from __future__ import annotations

import json
from pathlib import Path

from assumption_agent.benchmarks.sc100_shadow_oracle_qualification_v1 import (
    _CONTAINER_PROGRAM,
    _resolved_fixtures,
)


PROJECT = Path(__file__).parents[1]
FIXTURES = PROJECT / "manifests" / "sc100_shadow_oracle_qualification_fixtures_v1.json"


def test_fixture_pack_resolves_to_two_canaries_and_five_mutants() -> None:
    specification = json.loads(FIXTURES.read_text(encoding="utf-8"))
    blank, rows = _resolved_fixtures(PROJECT, specification)
    assert blank.name == "sc100-blank.pdf"
    assert len(rows) == 7
    assert sum(row["kind"] == "positive_canary" for row in rows) == 2
    assert sum(row["kind"] == "mutant" for row in rows) == 5
    assert all(len(row["filled_sha256"]) == 64 for row in rows)
    assert all(len(row["semantic_gold_sha256"]) == 64 for row in rows)


def test_gold_reference_patches_do_not_mutate_parent_canaries() -> None:
    specification = json.loads(FIXTURES.read_text(encoding="utf-8"))
    _, rows = _resolved_fixtures(PROJECT, specification)
    by_id = {row["fixture_id"]: row for row in rows}
    assert by_id["canary_q9_yes_q10_yes"]["semantic_gold"]["claim"]["amount"] == "3745"
    assert by_id["mutant_wrong_amount_gold"]["semantic_gold"]["claim"]["amount"] == "3746"
    assert by_id["canary_q9_no_q10_no"]["semantic_gold"]["claim"]["amount"] == "1675"
    assert by_id["mutant_q10_boundary_gold"]["semantic_gold"]["claim"]["amount"] == "2501"


def test_container_program_reads_only_mounted_conformance_inputs() -> None:
    assert "/oracle/sc100_shadow_oracle_v1.py" in _CONTAINER_PROGRAM
    assert "/fixtures/blank.pdf" in _CONTAINER_PROGRAM
    assert "/fixtures/filled.pdf" in _CONTAINER_PROGRAM
    assert "/fixtures/gold.json" in _CONTAINER_PROGRAM
    assert "synthetic_sc100_shadow_v1" not in _CONTAINER_PROGRAM
    assert "prompts/" not in _CONTAINER_PROGRAM
    assert "ruoli" not in _CONTAINER_PROGRAM.casefold()
