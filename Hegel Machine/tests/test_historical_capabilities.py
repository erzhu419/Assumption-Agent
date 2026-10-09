"""Historical routes execute the retained researchers rather than old snapshots."""

import json
from pathlib import Path

from hegel_machine.cognition.historical_capabilities import (
    execute_historical, historical_capabilities,
)


def test_every_historical_number_has_an_existing_current_entry():
    catalogue = historical_capabilities()
    assert {entry["version"] for entry in catalogue["versions"]} == {
        f"0.{version}" for version in range(1, 182)}
    root = Path(__file__).resolve().parents[1]
    for entry in catalogue["versions"]:
        assert entry["entry_points"] and entry["native_commands"]
        assert all((root / path).is_file() for path in entry["current_modules"])
    assert historical_capabilities(version="0.1")["versions"][0]["disposition"] == "legacy_protocol"
    withdrawn = historical_capabilities(version="0.174")["versions"][0]
    assert withdrawn["disposition"] == "reverted"
    assert withdrawn["native_commands"] == ["unified"]


def test_representation_route_runs_structural_discovery_and_saves_real_knowledge(tmp_path, capsys):
    output, memory = tmp_path / "structure.json", tmp_path / "knowledge.json"
    assert execute_historical(["structure", "--readout", "total", "--seed", "71",
        "--output", str(output), "--memory", str(memory)], group="representation") == 0
    result = json.loads(output.read_text())
    knowledge = json.loads(memory.read_text())
    assert result["costs"]["observation_cost"] > 0
    assert result["costs"]["compute_cost"] > 0
    assert knowledge["concepts"]


def test_late_history_route_uses_current_session_and_reopens_checkpoint(tmp_path, capsys):
    request, checkpoint, memory = (tmp_path / name for name in ("request.json", "session.json", "memory.json"))
    request.write_text(json.dumps({"config": {"seed": 71, "observation_budget": 96,
        "compute_budget": 120000, "minimum_improvement": 1e-7, "max_rounds": 2},
        "measurement_budget": 384,
        "tasks": [{"task_index": 0, "seed": 71}, {"task_index": 0, "seed": 91}]}))
    assert execute_historical(["unified", "--request", str(request), "--checkpoint", str(checkpoint),
        "--memory", str(memory), "--steps", "1"], version="0.181.0") == 0
    first = json.loads(checkpoint.read_text())
    assert len(first["results"]) == 1 and first["knowledge"]["concepts"]
    assert execute_historical(["unified", "--checkpoint", str(checkpoint), "--memory", str(memory),
        "--steps", "1"], version="0.181") == 0
    second = json.loads(checkpoint.read_text())
    assert len(second["results"]) == 2
    assert second["results"][0] == first["results"][0]
    assert json.loads(memory.read_text())["concepts"] == second["knowledge"]["concepts"]
