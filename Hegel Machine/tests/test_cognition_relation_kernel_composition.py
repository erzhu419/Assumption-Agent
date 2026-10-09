"""A learned relation can support a new response without copying its old wrapper."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.relation_kernel_composition import (
    collect_relation_kernels, compose_relation_kernels,
)
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch, _bind
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_relation_kernel_transfer import _focus_relation
from test_cognition_retained_relation_intervals import task


def _consumer(kernel, mode="event", *, gain=-1.7, intercept=.43):
    term = kernel
    if mode in {"event", "window_event"}:
        term = Program("event_delta", lag=1, children=(term,))
    if mode == "window_event":
        term = Program("window_sum", lag=3, children=(term,))
    return Program("add", children=(Program("const", value=intercept),
        Program("mul", children=(Program("const", value=gain), term))))


def _rows(program, seed, lengths=(23, 29), distances=(.18, .44, .78, 1.1)):
    rng, rows = Random(seed), []
    for length in lengths:
        history = []
        for index in range(length):
            focus = float(rng.randrange(2))
            origin = rng.uniform(-.4, .4)
            values = dict(f0=focus, f1=1. - focus, x0=origin,
                x1=origin + distances[index % len(distances)],
                q0=rng.uniform(-2., 2.), q1=rng.uniform(-2., 2.))
            if "unrelated" in program.required_features():
                values["unrelated"] = rng.uniform(-2., 2.)
            row = Observation(values, 0., tuple(history))
            rows.append(replace(row, target=program.predict(row)))
            history.append(dict(values))
    return rows


def _mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def _joint_role_target(*, window=False, quantity_gain=.8, event_gain=-1.3):
    quantity = Program("object_sum", children=tuple(
        Program("feature", name=f"q{index}") for index in range(2)))
    event = Program("event_delta", lag=1, children=(_focus_relation(.6),))
    combined = Program("add", children=(
        Program("mul", children=(Program("const", value=quantity_gain), quantity)),
        Program("mul", children=(Program("const", value=event_gain), event))))
    if window:
        combined = Program("window_sum", lag=3, children=(combined,))
    return Program("add", children=(Program("const", value=.23), combined))


def _role_task(program):
    objects = ["left", "right"]
    schema = {"objects": objects,
        "fields": {group: {role: f"{role}{index}" for role in ("q", "x", "f")}
                   for index, group in enumerate(objects)},
        "members": {group: [group] for group in objects},
        "controls": {group: {role: f"{role}{index}" for role in ("q", "x")}
                     for index, group in enumerate(objects)},
        "focus_control": "focus"}
    return replace(task(program), context={"object_schema": json.dumps(schema)})


@pytest.fixture(scope="module")
def wrapped_source():
    kernel = _focus_relation(.6)
    nuisance = Program("feature", name="unrelated")
    source = Program("add", children=(
        _consumer(kernel, "current", gain=2.1, intercept=-.37),
        Program("mul", children=(Program("const", value=-.8),
                                 Program("mul", children=(nuisance, nuisance))))))
    # The source actually predicts this complete response. Its saved relation
    # interval is obtained by running the existing research procedure on it.
    paid = _rows(source, 811)
    research = RelationIntervalResearch(source, [], SearchBudget(100000),
        structure={"fits": [{"radius": .6}], "radius_interval": [.5999, .6001]})
    alternatives = research.alternatives(paid, SearchBudget(100000))
    assert alternatives and all(_mse(item["candidate"].program, paid) < 1e-12 for item in alternatives)
    prior = json.loads(json.dumps({"source_concept_id": "wrapped_source", "status": "supported",
                                  "research_program": alternatives[0]["research_program"]}))
    return source, prior


@pytest.mark.parametrize("mode", ["current", "event", "window_event"])
def test_only_relation_is_transferred_and_current_response_is_refitted(wrapped_source, mode):
    source, prior = wrapped_source
    before = deepcopy(prior)
    actual = _consumer(_focus_relation(.6), mode)
    spec = task(actual)
    assert "unrelated" in source.required_features() - set(spec.feature_names)
    budget = SearchBudget(100000)
    records = collect_relation_kernels(spec, [prior], budget)
    assert records and all(kernel.op == "neighbor_sum" for kernel in records)
    assert all("unrelated" not in kernel.required_features() for kernel in records)
    claims = compose_relation_kernels(spec, _rows(actual, 113), budget, records)
    assert claims
    best = min(claims, key=lambda item: item["training_loss"])
    assert _mse(best["candidate"].program, _rows(actual, 331, (2, 31, 19))) < 1e-10
    assert best["candidate"].parents == ("wrapped_source",)
    assert best["method_source_ids"] == ["wrapped_source"]
    saved = best["research_program"]["selection_steps"][-1]
    assert saved["operation"] == "resolve_relation_intervals"
    assert saved["parameters"][0]["kernel_source_ids"] == ["wrapped_source"]
    assert 0 < budget.spent <= budget.limit
    assert prior == before


def test_kernel_only_meta_prior_enters_unified_response_research(wrapped_source):
    _, prior = wrapped_source
    actual = _consumer(_focus_relation(), "event")
    training, validation = _rows(actual, 512), _rows(actual, 813, (17, 11))
    library = KnowledgeLibrary()
    library.meta_priors.append(deepcopy(prior))
    config = ResearchConfig(compute_budget=120000, max_history=3, max_question_depth=2)
    agent = UnifiedResearchAgent(config, library=library)
    assert not library.concepts and agent._recall(task(actual)) == []
    actions, budget = [], SearchBudget(config.compute_budget)
    claims = agent._research_view(task(actual), training, validation, budget,
        actions, [], ResearchGraph(), observation_remaining=48)
    best = agent._best(claims)
    assert best is not None and best["validation_loss"] < 1e-10
    assert best["method_source_ids"] == ["wrapped_source"]
    assert any(action["action"] == "compose_relation_kernels" for action in actions)
    assert any(action["action"] == "choose_research_action"
               and action["selected_action"] == "program_revision" for action in actions)
    control = UnifiedResearchAgent(config, library=KnowledgeLibrary())
    without = control._research_view(task(actual), training, validation, SearchBudget(config.compute_budget),
        [], [], ResearchGraph(), observation_remaining=48)
    assert control._best(without)["validation_loss"] > 1e-5


def test_json_replay_refits_the_new_consumer_and_preserves_relation_parameters(wrapped_source):
    _, prior = wrapped_source
    first = _consumer(_focus_relation(), "window_event")
    budget = SearchBudget(100000)
    records = collect_relation_kernels(task(first), [prior], budget)
    claims = compose_relation_kernels(task(first), _rows(first, 513), budget, records)
    best = min(claims, key=lambda item: item["training_loss"])
    saved = json.loads(json.dumps(best["research_program"]))
    before = deepcopy(saved)
    changed = _consumer(_focus_relation(), "window_event", gain=2.7, intercept=-.81)
    replay_budget = SearchBudget(5000)
    replayed = replay_research_program(saved, _rows(changed, 921), replay_budget)
    assert replayed is not None
    fresh = _rows(changed, 8103, (3, 27))
    assert _mse(best["candidate"].program, fresh) > .1
    assert _mse(replayed["candidate"].program, fresh) < 1e-10
    step = replayed["research_program"]["selection_steps"][-1]
    assert step["operation"] == "resolve_relation_intervals"
    assert step["parameters"] == saved["selection_steps"][-1]["parameters"]
    rebound = _bind(Program.from_dict(step["parameter_template"]), {
        item["symbol"]: Program("const", value=item["radius"]) for item in step["parameters"]})
    assert _mse(rebound, fresh) < 1e-10
    assert 0 < replay_budget.spent <= replay_budget.limit
    assert saved == before


@pytest.mark.parametrize("window", [False, True])
def test_raw_relation_kernel_combines_with_measured_roles_across_new_trajectories(wrapped_source, window):
    _, prior = wrapped_source
    before = deepcopy(prior)
    actual = _joint_role_target(window=window)
    spec = _role_task(actual)
    budget = SearchBudget(120000)
    records = collect_relation_kernels(spec, [prior], budget)
    # These are raw learned relation subtrees, not Program.call wrappers.
    assert records and all(kernel.op == "neighbor_sum" for kernel in records)
    claims = compose_relation_kernels(spec, _rows(actual, 2311), budget, records)
    best = min(claims, key=lambda item: item["training_loss"])
    future = _rows(actual, 7913, lengths=(1, 2, 37, 51))
    assert _mse(best["candidate"].program, future) < 1e-10
    assert best["method_source_ids"] == ["wrapped_source"]
    assert best["candidate"].parents == ("wrapped_source",)
    assert any(step["operation"] in {"joint_role_residual", "derive_temporal_mechanism"}
               for step in best["research_program"]["selection_steps"])
    parameter = best["research_program"]["selection_steps"][-1]["parameters"][0]
    assert parameter["kernel_source_ids"] == ["wrapped_source"]
    assert prior == before
    assert 0 < budget.spent <= budget.limit


def test_unsolved_kernel_alternatives_do_not_prevent_saved_joint_procedure_replay(wrapped_source):
    _, prior = wrapped_source
    original = _joint_role_target(window=True)
    spec = _role_task(original)
    source_rows = _rows(original, 2917)
    budget = SearchBudget(120000)
    records = collect_relation_kernels(spec, [prior], budget)
    proposals = compose_relation_kernels(spec, source_rows, budget, records)
    best = min(proposals, key=lambda item: item["training_loss"])
    assert best["training_loss"] < 1e-10
    library = KnowledgeLibrary()
    source_id, added = library.learn(best["candidate"], spec, source_rows)
    assert added
    library.meta_priors.extend([deepcopy(prior), {
        "source_concept_id": source_id, "status": "supported",
        "object_schema": json.loads(spec.context["object_schema"]),
        "research_program": json.loads(json.dumps(best["research_program"]))}])
    agent = UnifiedResearchAgent(ResearchConfig(max_question_depth=2), library=library)
    changed = _joint_role_target(window=True, quantity_gain=-.4, event_gain=2.1)
    training = _rows(changed, 8291)
    actions, budget = [], SearchBudget(20000)
    claims = agent._program_claims(spec, training, budget, agent._recall(spec), actions)
    kernel_screen = next(action for action in actions if action["action"] == "compose_relation_kernels")
    assert len(kernel_screen["alternatives"]) >= 4
    assert all(item["training_loss"] > 1e-5 for item in kernel_screen["alternatives"])
    replayed = [claim for claim in claims if claim["research_program"].get("reused_procedure")]
    assert any(claim["method_source_ids"] == [source_id] for claim in replayed)
    future = _rows(changed, 7321, lengths=(2, 39))
    assert min(_mse(claim["candidate"].program, future) for claim in replayed) < 1e-10
    assert 0 < budget.spent <= budget.limit


def test_wrong_relation_cannot_explain_current_response(wrapped_source):
    _, prior = wrapped_source
    actual = _consumer(_focus_relation(.6), "event")
    wrong = deepcopy(prior)
    parameter = wrong["research_program"]["selection_steps"][-1]["parameters"][0]
    parameter.update(radius=.3, radius_intervals=[[.3, .3]], prior_radius_intervals=[[.3, .3]])
    budget = SearchBudget(50000)
    records = collect_relation_kernels(task(actual), [wrong], budget)
    claims = compose_relation_kernels(task(actual), _rows(actual, 817), budget, records)
    assert claims and min(item["training_loss"] for item in claims) > 1e-5
    assert min(_mse(item["candidate"].program, _rows(actual, 9817)) for item in claims) > 1e-5
    assert compose_relation_kernels(task(actual), _rows(actual, 817), SearchBudget(50000), {}) == []
    assert all(item["research_program"]["status"] == "requires_fresh_validation" for item in claims)


def test_unavailable_revoked_or_unfunded_source_never_creates_a_claim(wrapped_source):
    _, prior = wrapped_source
    before = deepcopy(prior)
    actual = _consumer(_focus_relation())
    missing = replace(task(actual), feature_names=tuple(name for name in task(actual).feature_names if name != "q1"))
    assert collect_relation_kernels(missing, [prior], SearchBudget(10000)) == {}
    for status in ("revoked", "counterexample", "proposed"):
        assert collect_relation_kernels(task(actual), [{**deepcopy(prior), "status": status}], SearchBudget(10000)) == {}
    tiny = SearchBudget(1)
    assert collect_relation_kernels(task(actual), [prior], tiny) == {}
    assert tiny.spent <= tiny.limit
    records = collect_relation_kernels(task(actual), [prior], SearchBudget(10000))
    unfunded = SearchBudget(1)
    assert compose_relation_kernels(task(actual), _rows(actual, 218), unfunded, records) == []
    assert unfunded.spent <= unfunded.limit
    assert prior == before


def test_quiet_data_keeps_both_source_intervals_executable_after_json(wrapped_source):
    _, original = wrapped_source
    priors = []
    for source_id, radius, interval in (("short_range", .4, [.39, .41]),
                                        ("long_range", .8, [.79, .81])):
        prior = deepcopy(original)
        prior["source_concept_id"] = source_id
        prior["research_program"]["selection_steps"][-1]["parameters"][0].update(
            radius=radius, radius_intervals=[interval], prior_radius_intervals=[interval])
        priors.append(prior)
    actual = _consumer(_focus_relation(.6), "event")
    quiet = _rows(actual, 391, distances=(.1, 1.2))
    budget = SearchBudget(100000)
    records = collect_relation_kernels(task(actual), priors, budget)
    claims = compose_relation_kernels(task(actual), quiet, budget, records)
    assert claims
    selected = min(claims, key=lambda item: item["training_loss"])
    saved = json.loads(json.dumps(selected["research_program"]))
    before = deepcopy(saved)
    parameter = saved["selection_steps"][-1]["parameters"][0]
    assert parameter["radius_intervals"] == [[.39, .41], [.79, .81]]
    assert {item["source_concept_id"] for item in parameter["kernel_priors"]} == {"short_range", "long_range"}
    assert len(parameter["kernel_source_ids"]) == 1
    assert selected["method_source_ids"] == parameter["kernel_source_ids"]
    restored = RelationIntervalResearch(selected["candidate"].program, [], SearchBudget(100000),
                                       research_program=saved)
    alternatives = restored.alternatives(quiet, SearchBudget(100000))
    assert len(alternatives) >= 2 and restored.questions(alternatives)
    assert all(_mse(entry["candidate"].program, quiet) < 1e-10 for entry in alternatives)
    boundary = _rows(actual, 714, (13,), distances=(.1, .6, 1.2))
    predictions = [[entry["candidate"].program.predict(row) for row in boundary] for entry in alternatives]
    assert max(max(values) - min(values) for values in zip(*predictions)) > .1
    assert saved == before
