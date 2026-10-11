"""An exhausted reference search can compose its actual prior intermediate terms."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import world_reference_expansion as expansion
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from test_world_reference_expansion import PATH, retained, setup_agent


def prepared(tmp_path, feature="drive"):
    term = Program("feature", name=feature)
    square = Program("mul", children=(term, term))
    old = [{"id": "original_linear", "basis": [term.to_dict()]},
           {"id": "original_squared", "basis": [term.to_dict(), square.to_dict()]}]
    rows = [{"crystal": i, "measurements": {feature: {"value": float(i), "evidence": i}},
             **({"frequency": {"value": float(100 + 10 * i ** 3), "evidence": i}}
                if i < 4 else {})} for i in range(1, 5)]
    agent = setup_agent(tmp_path, old, {"references": rows[:3], "targets": rows[3:]})
    state = json.loads((tmp_path / "shared/reference_recalibration.json").read_text())
    state.update(expanded_features=[feature], response_settings=[
        {"crystal": i, "frequency": float(100 + 10 * i ** 3 + offset),
         "activated": offset == 0, "evidence": [10 * i + offset]}
        for i in range(1, 4) for offset in (-1, 0, 1)],
        response_research={"status": "family_research_needed", "family_research": {
            "research_obligation": {"reason": "declared_shared_response_families_refuted"}},
            "basis_expansion_action": {"action": "run", "arguments": {
                "operator": "reference_expansion", "path": PATH}}})
    save(agent, state)
    return agent, state, old


def save(agent, state):
    (agent.workspace / "shared/reference_recalibration.json").write_text(json.dumps(state))


def load(agent):
    return json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())


@pytest.mark.parametrize("feature", ["drive", "renamed_reading"])
def test_saved_intermediate_is_composed_and_checked_on_all_paid_responses(tmp_path, feature):
    agent, before, original = prepared(tmp_path, feature)
    result = expansion.execute_reference_expansion(agent, object(), {"path": PATH})
    generated = result["generated_hypotheses"]
    assert generated
    terms = [Program.from_dict(term) for item in generated for term in item["basis"]]
    assert any(all(term.predict(Observation({feature: value}, 0.)) == pytest.approx(value ** 3)
                   for value in (-2., -.5, 1., 2., 3.)) for term in terms)
    after = load(agent)
    assert list(retained(agent)[:len(original)]) == original
    assert after["research_table"] == before["research_table"]
    assert after["response_settings"] == before["response_settings"]
    assert "basis_expansion_action" not in after["response_research"]
    assert result["composition_attempts"][0]["source_ids"] == ["original_linear", "original_squared"]
    for declaration in generated:
        check = after["family_history_checks"][declaration["id"]]
        assert check["paid_response_count"] == len(before["response_settings"])
        assert check["responses"] == before["response_settings"] and check["queries"] == []
        assert check["basis"] == declaration["basis"]
    assert agent.costs["research_compute_units"] == 7 + result["compute_cost"]


def test_generated_candidates_do_not_feed_an_unbounded_polynomial_tower(tmp_path):
    agent, _, _ = prepared(tmp_path)
    first = expansion.execute_reference_expansion(agent, object(), {"path": PATH})
    source, state = (agent.workspace / PATH).read_text(), load(agent)
    assert first["generated_hypotheses"]
    assert not expansion.pending_reference_compositions(state, retained(agent))
    again = expansion.execute_reference_expansion(agent, object(), {"path": PATH})
    assert again["compute_cost"] == 0 and again["generated_hypotheses"] == []
    assert (agent.workspace / PATH).read_text() == source and load(agent) == state
    # Another point outcome does not create a new structural seed.
    state["response_settings"].append({"crystal": 4, "frequency": 500.,
                                      "activated": False, "evidence": [51]})
    assert not expansion.pending_reference_compositions(state, retained(agent))
    # A later author-supplied coordinate does create a distinct opportunity.
    term = Program("feature", name="drive")
    fourth = Program("mul", children=(Program("mul", children=(term, term)),
                                      Program("mul", children=(term, term))))
    authored = [*retained(agent), {"id": "authored_fourth", "basis": [fourth.to_dict()]}]
    assert expansion.pending_reference_compositions(state, authored)


def test_paid_impossibility_is_retained_for_every_generated_family(tmp_path):
    agent, state, original = prepared(tmp_path)
    # Same object has a negative between two positive settings: no connected
    # activation band in any generated coordinate can explain these paid data.
    state["response_settings"] = [{"crystal": 1, "frequency": frequency,
        "activated": frequency != 2., "evidence": [i + 20]}
        for i, frequency in enumerate((1., 2., 3.))]
    state["family_history_checks"] = {"old_evidence": {"status": "UNKNOWN"}}
    save(agent, state)
    result = expansion.execute_reference_expansion(agent, object(), {"path": PATH})
    assert result["paid_family_checks"]
    assert all(check["status"] == "UNSAT_VERIFIED" and check["conflicting_conditions"]
               for check in result["paid_family_checks"].values())
    after = load(agent)
    assert after["family_history_checks"]["old_evidence"] == {"status": "UNKNOWN"}
    assert list(retained(agent)[:len(original)]) == original
    assert after["response_settings"] == state["response_settings"]


def test_composition_requires_the_recorded_scientific_obligation(tmp_path):
    agent, state, _ = prepared(tmp_path)
    for field in ("status", "family_research"):
        modified = deepcopy(state)
        modified["response_research"].pop(field)
        assert not expansion.pending_reference_compositions(modified, retained(agent))
