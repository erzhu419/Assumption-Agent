"""Paid full-family exclusions cannot buy a new old-member exploration epoch."""
from copy import deepcopy

import pytest

from hegel_machine.cognition import world_response_consequences as consequences
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.world_response_constraints import query_response_family
from test_world_response_consequences import (
    LINEAR, QUADRATIC, assess, experiment, fixture, remember, response, selected_result,
)

CONSTANT = {"id": "constant", "basis": [Program("const", value=1.).to_dict()]}


def stalled_state():
    state = fixture()
    state.pop("consequence_panel")
    # Equal constant values cannot explain 10/12 positives and the paid 11 negative.
    state["response_settings"].append(response(1, 11., False))
    first = assess(state, [LINEAR], [experiment("old_region", 3, 14.)])
    remember(state, first)
    state["response_research"]["family_research"]["no_progress_cycles"] = 3
    return state


def features(state):
    table = state["research_table"]
    return {row["crystal"]: {name: value["value"] for name, value in row["measurements"].items()}
            for row in table["references"] + table["targets"]}


def excluded(state):
    answer = query_response_family(features(state), state["response_settings"], CONSTANT)
    assert answer["status"] == "UNSAT_VERIFIED"
    return answer


@pytest.mark.parametrize("source", ["submission", "retained"])
def test_excluded_new_family_preserves_stagnation_and_original_panel(source):
    state = stalled_state()
    answer = excluded(state)
    paid = None
    if source == "submission":
        paid = {CONSTANT["id"]: answer}
    else:
        state["family_history_checks"] = {CONSTANT["id"]: {
            **answer, "basis": deepcopy(CONSTANT["basis"]), "queries": [],
            "features": {str(key): value for key, value in features(state).items()},
            "responses": deepcopy(state["response_settings"]),
        }}
    before = deepcopy(state["response_settings"])
    result = consequences.assess_response_consequences(
        state, [LINEAR, CONSTANT], selected_result([experiment("other_region", 3, 16.)]),
        paid_checks=paid)
    research = result["family_research"]
    assert research["no_progress_cycles"] == 3
    assert research["panel"] == [{"crystal": 3, "frequency": 14.}]
    assert research["epoch"] == state["response_research"]["family_research"]["epoch"]
    assert research["nonlocal_progress"] == 0
    assert next(item for item in research["families"] if item["id"] == CONSTANT["id"])["status"] == "UNSAT_VERIFIED"
    assert CONSTANT["basis"] in research["bases"]
    assert state["response_settings"] == before
    # The newly authored, excluded family remains recorded, without renewing this old choice.
    assert result["status"] == "family_research_needed"
    assert result["selection"] is None


def test_new_excluded_family_cannot_hide_completion_of_the_old_selected_test():
    state = stalled_state()
    answer = excluded(state)
    state["response_settings"].append(response(3, 14., False))
    result = consequences.assess_response_consequences(
        state, [LINEAR, CONSTANT], selected_result([experiment("next_region", 3, 16.)]),
        paid_checks={CONSTANT["id"]: answer})
    assert result["family_research"]["no_progress_cycles"] == 4


@pytest.mark.parametrize("new_status", ["SAT_WITNESS", "UNKNOWN"])
def test_live_new_family_retains_its_own_opportunity_with_excluded_addition(new_status):
    state = stalled_state()
    answer = excluded(state)
    extra = query_response_family(features(state), state["response_settings"], QUADRATIC,
                                  compute_limit=1 if new_status == "UNKNOWN" else 50000)
    assert extra["status"] == new_status
    result = consequences.assess_response_consequences(
        state, [LINEAR, CONSTANT, QUADRATIC],
        selected_result([experiment("new_region", 3, 16.)]),
        paid_checks={CONSTANT["id"]: answer, QUADRATIC["id"]: extra})
    research = result["family_research"]
    assert research["no_progress_cycles"] == 0
    assert research["panel"] == [{"crystal": 3, "frequency": 16.}]
    assert research["epoch"] == state["response_research"]["family_research"]["epoch"] + 1
    assert "research_obligation" not in research
