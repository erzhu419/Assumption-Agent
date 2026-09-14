import json
from random import Random

import pytest

from hegel_machine.cognition.continuous_state import _linear_fit, _sum_terms
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.research_programs import (
    propose_research_programs, replay_research_program,
)


def source_calls():
    """Fit two source-task computations, then preserve them as callable atoms."""
    rng = Random(21)
    source = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(30)]
    columns = [[point[index] for point in source] for index in range(2)]
    total = _linear_fit(columns, [left + right for left, right in source])
    body = _sum_terms([Program("arg", name="left"), Program("arg", name="right")],
                      total[1:], total[0])
    definition = Definition("learned_total", ("left", "right"), body, "concept_total")
    aggregate = Program("call", name=definition.definition_id, definition=definition,
        children=(Program("feature", name="a"), Program("feature", name="b")))
    event_source = [rng.uniform(-1, 1) for _ in range(30)]
    differences = [event_source[index] - event_source[index - 1]
                   for index in range(1, len(event_source))]
    event_fit = _linear_fit([differences], [2. * value for value in differences])
    event = Definition("learned_event", ("input",),
        _sum_terms([Program("event_delta", lag=1, children=(Program("arg", name="input"),))],
                   event_fit[1:], event_fit[0]), "concept_event")
    change = Program("call", name=event.definition_id, definition=event,
                     children=(Program("feature", name="u"),))
    return aggregate, change


def evidence(seed, *, mode="add", gains=(1.3, -.7), lengths=(17, 23)):
    rng = Random(seed)
    rows = []
    for length in lengths:
        history, values, previous, state = [], [], None, 0.
        for _ in range(length):
            features = {name: rng.uniform(-1, 1) for name in ("a", "b", "u")}
            total = features["a"] + features["b"]
            event = 0. if previous is None else 2. * (features["u"] - previous)
            combined = gains[0] * total + gains[1] * event
            values.append(combined)
            state = .8 * state + combined
            target = {"add": combined, "window": sum(values[-3:]),
                      "state": state, "product": total * event}[mode] + .23
            rows.append(Observation(features, target, tuple(history)))
            history.append(dict(features))
            previous = features["u"]
    return rows


def spec():
    return TaskSpec("new_task", "Observe the apparatus", ("a", "b", "u"), (), ())


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


@pytest.mark.parametrize("mode", ["add", "window", "state", "product"])
def test_learned_components_explain_new_cross_structure_tasks(mode):
    rows = evidence(54, mode=mode)
    budget = SearchBudget(60000)
    proposals = propose_research_programs(spec(), rows, budget, source_calls())
    assert proposals
    best = min(proposals, key=lambda item: item["training_loss"])
    future = evidence(907, mode=mode, lengths=(3, 41, 55))
    assert mse(best["candidate"].program, future) < 1e-10
    assert set(best["candidate"].parents) == {"concept_total", "concept_event"}
    assert 0 < budget.spent <= budget.limit
    restored = json.loads(json.dumps(best["research_program"]))
    graph = ResearchGraph.from_dict(restored["graph"])
    assert mse(graph.nodes[restored["output_node"]].answer, future) < 1e-10
    assert graph.nodes["answer"].depends_on
    assert restored["status"] == "requires_fresh_validation"


def test_saved_research_program_refits_nested_calls_after_json_roundtrip():
    initial = evidence(63, mode="window")
    proposals = propose_research_programs(spec(), initial, SearchBudget(60000), source_calls())
    first = min(proposals, key=lambda item: item["training_loss"])
    restored = json.loads(json.dumps(first["research_program"]))
    changed = evidence(101, mode="window", gains=(-.5, 1.8))
    budget = SearchBudget(1000)
    revised = replay_research_program(restored, changed, budget)
    assert revised is not None
    future = evidence(995, mode="window", gains=(-.5, 1.8), lengths=(5, 47))
    assert mse(first["candidate"].program, future) > .1
    assert mse(revised["candidate"].program, future) < 1e-10
    assert budget.spent < 1000
    assert revised["research_program"]["reused_procedure"]
    assert set(revised["candidate"].parents) == {"concept_total", "concept_event"}


def test_missing_or_uninformative_components_and_unfunded_search_add_no_claim():
    rows = evidence(76)
    budget = SearchBudget(10000)
    missing = source_calls()[0].remap_features({"a": "absent"})
    assert propose_research_programs(spec(), rows, budget, [missing]) == []
    assert budget.spent == 0
    constant = Definition("constant", (), Program("const", value=1.), "constant_concept")
    call = Program("call", name="constant", definition=constant)
    assert propose_research_programs(spec(), rows, budget, [call]) == []
    unfunded = SearchBudget(2 * len(rows))
    assert propose_research_programs(spec(), rows, unfunded, source_calls()) == []
    assert unfunded.spent == 0


@pytest.mark.parametrize("aggregate", [False, True])
def test_saved_research_program_binds_new_measurements_and_refits_changed_gains(aggregate):
    proposals = propose_research_programs(spec(), evidence(63, mode="window"),
        SearchBudget(60000), source_calls())
    first = min(proposals, key=lambda item: item["training_loss"])
    saved = json.loads(json.dumps(first["research_program"]))
    mapping = {"a": "quantity", "b": "other_quantity", "u": "signal"}
    if aggregate:
        mapping["a"] = Program("object_sum", children=(
            Program("feature", name="quantity_left"), Program("feature", name="quantity_right")))

    def measured(rows):
        def rename(features):
            result = {"other_quantity": features["b"], "signal": features["u"]}
            if aggregate:
                result.update(quantity_left=.3 * features["a"], quantity_right=.7 * features["a"])
            else:
                result["quantity"] = features["a"]
            return result
        return [Observation(rename(row.features), row.target,
                            tuple(rename(point) for point in row.history)) for row in rows]

    budget = SearchBudget(1000)
    changed = measured(evidence(101, mode="window", gains=(-.5, 1.8)))
    result = replay_research_program(saved, changed, budget, feature_mapping=mapping)
    assert result is not None
    future = measured(evidence(995, mode="window", gains=(-.5, 1.8), lengths=(5, 47)))
    assert mse(result["candidate"].program, future) < 1e-10
    assert result["candidate"].program.required_features() == set(future[0].features)
    assert budget.spent < 1000
