"""Measured child answers are executed by a changeable dependency graph."""

import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph


def argument(name):
    return Program("arg", name=name)


def product(left, right):
    return Program("mul", children=(left, right))


def restore(graph):
    return ResearchGraph.from_dict(json.loads(json.dumps(graph.to_dict())))


def observations(coefficient, names=("x",)):
    return [Observation({names[0]: x}, coefficient * x) for x in (-2., -1., 0., 1., 2.)]


def test_three_levels_consume_measured_answers_and_resume_after_budget_exhaustion():
    graph = ResearchGraph()
    graph.add_question("gain", "Learn the measured input response", operations=("refit",),
                       observations=observations(2.))
    graph.add_question("interaction", "Compose two uses of the input response", depends_on=("gain",),
                       expression=product(argument("gain"), argument("gain")))
    graph.add_question("prediction", "Add the dependent interaction and response", depends_on=("interaction", "gain"),
                       expression=Program("add", children=(argument("interaction"), argument("gain"))))
    inputs = [Observation({"x": x}, 4 * x * x + 2 * x) for x in (-1.8, -.7, .2, .8, 1.3)]
    partial = graph.execute_ready(inputs, SearchBudget(500), max_nodes=1)
    assert len(partial) == 1 and graph.nodes["gain"].status == "answered"
    assert loss(graph.nodes["gain"].answer, observations(2.), 0., .5) < 1e-12
    graph = restore(graph)
    assert graph.execute_ready(inputs, SearchBudget(0)) == []
    assert graph.nodes["gain"].answer is not None
    assert graph.nodes["interaction"].status == "paused"
    graph = restore(graph)
    budget = SearchBudget(100)
    result = graph.execute_ready(inputs, budget)
    final = next(candidate for candidate in result if candidate.candidate_id == "answer:prediction")
    assert loss(final.program, inputs, 0., .5) < 1e-12
    assert budget.spent == 2 * len(inputs)
    assert graph.nodes["gain"].answer in final.program.children
    assert graph.nodes["interaction"].answer in final.program.children
    assert graph.execute_ready(inputs, SearchBudget(100)) == []


def test_actual_missing_sensor_readings_block_and_then_unblock_a_dependent_question():
    graph = ResearchGraph()
    graph.add_question("instrument_response", "Explain the unmeasured driver", operations=("refit",),
                       required_measurements=("sensor_a",))
    graph.add_question("prediction", "Execute the measured response", depends_on=("instrument_response",),
                       expression=argument("instrument_response"))
    controls = observations(2.)
    budget = SearchBudget(500)
    assert graph.execute_ready(controls, budget) == []
    assert budget.spent == 0
    assert graph.measurement_requests()[0]["measurements"] == ["sensor_a"]
    assert graph.supply_measurements(controls) == []
    graph = restore(graph)
    measured = observations(3., ("sensor_a",))
    assert graph.supply_measurements(measured) == ["instrument_response"]
    assert graph.measurement_requests() == []
    result = graph.execute_ready(measured, budget)
    assert {candidate.candidate_id for candidate in result} == {"answer:instrument_response", "answer:prediction"}
    assert loss(graph.nodes["prediction"].answer, measured, 0., .5) < 1e-12
    assert sum(node.compute_cost for node in graph.nodes.values()) == budget.spent <= budget.limit


def test_changing_dependencies_reopens_descendants_and_replaces_the_actual_computation():
    graph = ResearchGraph()
    graph.add_question("old_driver", "Fit the original component", operations=("refit",), observations=observations(2.))
    graph.add_question("new_driver", "Fit the revised component", operations=("refit",), observations=observations(3.))
    graph.add_question("sum", "Use the supported driver", depends_on=("old_driver",), expression=argument("old_driver"))
    graph.add_question("outer", "Use the dependent result", depends_on=("sum",), expression=argument("sum"))
    graph.execute_ready(observations(2.), SearchBudget(1000))
    previous = graph.nodes["outer"].answer
    assert loss(previous, observations(2.), 0., .5) < 1e-12
    graph.set_dependencies("sum", ("new_driver",), expression=argument("new_driver"))
    assert graph.nodes["sum"].answer is None and graph.nodes["outer"].answer is None
    assert previous in graph.nodes["outer"].alternatives
    graph = restore(graph)
    changed = graph.execute_ready(observations(3.), SearchBudget(100))
    assert changed and loss(graph.nodes["outer"].answer, observations(3.), 0., .5) < 1e-12
    assert loss(previous, observations(3.), 0., .5) > 1.
    graph.set_dependencies("sum", (), expression=Program("feature", name="x"))
    graph.execute_ready(observations(1.), SearchBudget(100))
    assert graph.nodes["sum"].depends_on == []
    assert loss(graph.nodes["outer"].answer, observations(1.), 0., .5) == 0.
    with pytest.raises(ValueError, match="themselves"):
        graph.set_dependencies("sum", ("outer",))


def test_dependency_program_discovers_its_required_measurement_before_execution():
    graph = ResearchGraph()
    graph.add_question("response", "Use the proposed instrument coordinate",
                       expression=Program("feature", name="sensor_a"))
    controls = observations(1.)
    budget = SearchBudget(100)
    assert graph.execute_ready(controls, budget) == []
    assert budget.spent == 0
    assert graph.nodes["response"].status == "blocked"
    assert graph.measurement_requests()[0]["measurements"] == ["sensor_a"]
    measured = observations(1., ("sensor_a",))
    graph.supply_measurements(measured)
    assert graph.execute_ready(measured, budget)
    assert loss(graph.nodes["response"].answer, measured, 0., .5) == 0.
