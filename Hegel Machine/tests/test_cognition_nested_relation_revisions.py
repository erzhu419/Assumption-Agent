"""A relation can change inside learned calls while their outer history survives."""

from copy import deepcopy
from dataclasses import replace
from random import Random

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_revisions import _difference, _expose, _geometry, _relations, propose_relation_revisions
from hegel_machine.cognition.unified_session import UnifiedResearchSession


def call(name, arguments, body, children):
    return Program("call", name=name, children=tuple(children),
        definition=Definition(name, tuple(arguments), body, name))


def composition(radius=.6, prefix=""):
    arg = lambda name: Program("arg", name=name)
    feature = lambda name: Program("feature", name=prefix + name)
    names = ("f0", "x0", "q0", "f1", "x1", "q1")
    relation = call("relation", names,
        Program("neighbor_sum", children=(Program("const", value=radius), *(arg(name) for name in names))),
        [feature(name) for name in names])
    total = call("total", ("left", "right"), Program("object_sum", children=(arg("left"), arg("right"))),
                 [feature("q0"), feature("q1")])
    event = call("event", ("value",), Program("event_delta", lag=1, children=(arg("value"),)), [relation])
    combined = call("combined", ("total", "change"), Program("add", children=(
        Program("mul", children=(Program("const", value=1.3), arg("total"))),
        Program("mul", children=(Program("const", value=-.7), arg("change"))))), [total, event])
    return call("window", ("value",), Program("window_sum", lag=3, children=(arg("value"),)), [combined])


def rows_for(program, seed, lengths=(16, 19, 23)):
    random, rows = Random(seed), []
    names = sorted(program.required_features())
    for length in lengths:
        history = []
        for _ in range(length):
            features = {name: random.uniform(-1., 1.) for name in names}
            for name in names:
                if name.endswith("f0"):
                    features[name] = 1.
                elif name.endswith("f1"):
                    features[name] = 0.
            row = Observation(features, 0., tuple(history))
            rows.append(replace(row, target=program.predict(row)))
            history.append(features)
    return rows


def mse(program, rows):
    return sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)


def test_nested_delta_preserves_lexical_arguments_window_and_event_startup_after_rename():
    old, changed = composition(), composition(1.2)
    original = deepcopy(old.to_dict())
    exposed = _expose(old, SearchBudget(2000))
    delta = _difference(exposed, _geometry(_relations(exposed)[0]), 1.2)
    revised = Program("add", children=(old, delta))
    rows = rows_for(changed, 17, (1, 2, 4, 27))
    assert mse(old, rows) > 1e-3
    assert mse(revised, rows) < 1e-24
    assert old.to_dict() == original
    assert {"total", "event", "combined", "window"} <= set(_sources(revised))
    renamed = revised.remap_features({name: "new_" + name for name in old.required_features()})
    assert mse(renamed, rows_for(composition(1.2, "new_"), 19, (1, 3, 31))) < 1e-24


def test_actual_rows_select_relation_cells_without_changing_the_retained_computation():
    old, changed = composition(), composition(1.2)
    training, validation = rows_for(changed, 31), rows_for(changed, 41, (29, 33))
    spec = TaskSpec("changed_relation", "", tuple(sorted(old.required_features())), (), ())
    budget = SearchBudget(16000)
    proposals = propose_relation_revisions(spec, training, budget, [old])
    assert len(proposals) >= 2
    selected = min(proposals, key=lambda item: mse(item["candidate"].program, validation))
    assert mse(selected["candidate"].program, validation) < 1e-7
    assert "window" in selected["candidate"].parents
    for item in proposals:
        step = item["research_program"]["selection_steps"][-1]
        assert step["radius_interval"][0] < step["radius"] < step["radius_interval"][1]
        assert item["research_program"]["status"] == "requires_fresh_validation"
    assert budget.spent <= 16000
    # No invented relation candidates or fits for a scalar mechanism.
    scalar = Program("mul", children=(Program("feature", name="q0"), Program("feature", name="q1")))
    ordinary_budget = SearchBudget(1000)
    assert not propose_relation_revisions(spec, training, ordinary_budget, [scalar])
    assert ordinary_budget.spent == 3
    assert not propose_relation_revisions(spec, training, SearchBudget(0), [old])


def test_nested_relation_uncertainty_is_researched_before_confirmation_and_old_call_remains_returnable():
    apparatus = {"environment": "objects", "objects": 4, "apparatus_seed": 9013}
    tasks = [{**apparatus, "readout": readout, "seed": seed} for readout, seed in (
        ("total", 10078), ("event", 20085), ("total_event", 30092), ("window_total_event", 50106))]
    tasks.extend([{**apparatus, "readout": "window_total_event", "seed": 60113, "relation_radius": 1.35},
                  {**apparatus, "readout": "window_total_event", "seed": 70120}])
    session = UnifiedResearchSession(ResearchConfig(observation_budget=384, compute_budget=360000,
        minimum_improvement=1e-7, max_rounds=6, active_probes=True), tasks=tasks, measurement_budget=4608)
    source = session.run(max_tasks=4)
    original_definitions = deepcopy(source["knowledge"]["definitions"])
    original_instruments = deepcopy(source["knowledge"]["instrument_models"])
    assert all(item["research"]["confirmation_loss"] < 1e-7 for item in source["results"])
    report = session.run()
    changed, returned = [item["research"] for item in report["results"][-2:]]
    assert changed["costs"]["measurement_cost"] == 0
    refinements = [action for action in changed["actions"] if action["action"] == "refine_relation_parameter_space"]
    probes = [action for action in changed["actions"]
              if action["action"] == "acquire" and action["purpose"] == "distinguish_program_conditions"]
    assert len(refinements) == len(probes) + 1 > 2
    assert refinements[-1]["evidence_rows"] == refinements[0]["evidence_rows"] + sum(
        action["observation_cost"] for action in probes)
    first_low, first_high = refinements[0]["after_intervals"][0]
    low, high = refinements[-1]["after_intervals"][0]
    assert low < 1.35 < high and high - low < (first_high - first_low) / 100
    assert all(action["selection_basis"] == "all_acquired_evidence" for action in refinements)
    confirmations = [action for action in changed["actions"]
                     if action["action"] == "validate_frozen_program" and action["accepted"]]
    assert changed["actions"].index(refinements[-1]) < changed["actions"].index(confirmations[-1])
    assert refinements[-1]["evidence_rows"] + confirmations[-1]["observation_rows"] <= changed["costs"]["observation_cost"]
    assert changed["research_program"]["status"] == "supported_on_fresh_trajectory"
    assert set(source["results"][3]["research"]["concepts_added"]) & set(changed["used_sources"])
    assert returned["costs"]["measurement_cost"] == 0
    assert returned["costs"]["observation_cost"] <= 384
    assert set(source["results"][3]["research"]["concepts_added"]) & set(
        [*returned["used_sources"], *returned.get("method_source_ids", [])])
    assert report["knowledge"]["instrument_models"] == original_instruments
    assert all(definition in report["knowledge"]["definitions"] for definition in original_definitions)
    for index in (4, 5):
        from hegel_machine.cognition.unified_session import laboratory_for
        with laboratory_for(tasks[index]) as lab:
            research = report["results"][index]["research"]
            program = Program.from_dict(research["program"])
            for length, seed in ((1, 1921), (3, 1922), (64, 92341)):
                observations = lab.acquire(lab.random_probe(seed, length=length), research["selected_channels"]).observations
                assert mse(program, observations) < 1e-7
                if index == 5:
                    original = Program.from_dict(source["results"][3]["research"]["program"])
                    assert mse(original, observations) < 1e-7
