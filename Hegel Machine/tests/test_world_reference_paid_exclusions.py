"""Retained full-family counterexamples must also constrain declaration trials."""

import json

import pytest

from hegel_machine.cognition import world_reference_hypothesis as entry
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.world_reference_response import execute_reference_response
from test_world_reference_hypothesis import PATH, _paid_shared_band_conflict, square


@pytest.mark.parametrize("change", ["unknown", "hypothetical", "signature", "basis", "measurement"])
def test_only_same_binding_paid_full_family_exclusions_are_reused(tmp_path, change):
    agent, lab = _paid_shared_band_conflict(tmp_path)
    path = agent.workspace / PATH
    source = path.read_text()
    for name in ("density", "gap"):
        source = entry._retain_declaration(source, square(name, name + "_square"))
    path.write_text(source)
    execute_reference_response(agent, lab, {"path": PATH})
    state = json.loads((agent.workspace / recipe.STATE).read_text())
    # Exercise the comparison's retained proof, independent of the same valid
    # exclusion that the ordinary comparison now also stores for future use.
    state['family_history_checks'] = {}
    family = next(item for item in state["response_research"]["family_research"]["families"]
                  if item["id"] == "density_square")
    assert family["status"] == "UNSAT_VERIFIED"
    if change == "unknown":
        family["status"] = "UNKNOWN"
    elif change == "hypothetical":
        family["queries"] = [{"crystal": 5, "frequency": 111., "activated": True}]
    elif change == "signature":
        state["response_research"]["family_research"]["families"].remove(family)
        state["response_research"]["family_research"]["signatures"].append(
            {"status": "UNSAT_VERIFIED", "family": "density_square"})
    elif change == "basis":
        family["basis"] = [{"op": "const", "value": 1.}]
    else:
        lab.density[2] = 2.
        agent._execute("query", {"actions": [{"action": "USE", "arg1": 91, "arg2": 202}]}, lab)
    before = len(agent.observations)
    reused = entry._reuse_paid_family_exclusions(agent, lab, state, source)
    assert "density_square" not in reused
    assert "gap_square" in reused
    assert len(agent.observations) == before
    assert agent.costs["model_calls"] == 0
