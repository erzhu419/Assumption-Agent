"""Control evidence can revise a retained mechanism before paying for meters."""

from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest

from hegel_machine.cognition.control_revisions import RetainedControlRevisions, _polynomial, _term
from hegel_machine.cognition.dynamic_instrument_lab import DynamicInstrumentLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_history=4, max_rounds=5, seed=113, active_probes=True)


@pytest.fixture(scope="module")
def source():
    library = KnowledgeLibrary()
    with DynamicInstrumentLab(seed=113, apparatus_seed=113) as lab:
        report = UnifiedResearchAgent(CONFIG, library=library, measurement_budget=1536).run(lab)
    assert report["confirmation_loss"] < 1e-7
    return library.to_dict(), report


def run_task(library, index, *, laboratory=DynamicInstrumentLab):
    seed = 113 + 10007 * index
    with laboratory(task_index=index, seed=seed, apparatus_seed=113) as lab:
        report = UnifiedResearchAgent(replace(CONFIG, seed=seed),
            library=library, measurement_budget=1536).run(lab)
        random = Random(931)
        rows = lab.acquire(Probe("independent", tuple({"u": random.uniform(-1., 1.)}
            for _ in range(64))), tuple(report["selected_channels"])).observations
    program = Program.from_dict(report["program"])
    assert sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows) < 1e-7
    return report


def test_changed_delay_and_mechanism_use_control_evidence_and_preserve_old_instrument_knowledge(source):
    library = KnowledgeLibrary.from_dict(source[0])
    old_instruments = deepcopy(library.instrument_models)
    old_definitions = {name: definition.to_dict() for name, definition in library.definitions.items()}
    changed = run_task(library, 2)
    assert changed["selected_kind"] == "control_history_revision"
    assert changed["costs"]["measurement_cost"] == 0
    assert not any(action["action"] == "instrument_mapping_counterexample" for action in changed["actions"])
    revised = run_task(library, 3)
    assert revised["selected_kind"] == "control_mechanism_refit"
    assert revised["costs"]["measurement_cost"] == 0
    assert library.instrument_models == old_instruments
    returned = run_task(library, 4)
    assert returned["selected_kind"] == "retained_calibrated_call"
    assert returned["costs"]["observation_cost"] == 24
    assert returned["costs"]["measurement_cost"] == 0
    assert {name: library.definitions[name].to_dict() for name in old_definitions} == old_definitions
    for report in (changed, revised):
        trial = next(action for action in report["actions"]
            if action["action"] == "confirm_observation_research"
            and action["research_action"] == "revise_mechanism" and action["accepted"])
        assert trial["observation_rows"] > 0 and trial["after_loss"] < 1e-7
        choice = next(action for action in report["actions"]
                      if action["action"] == "choose_observation_research")
        assert choice["selected_action"] == "revise_mechanism"
        assert choice["control_revision_applicability"]["observation_rows"] >= 16


class AbsoluteInstrument(DynamicInstrumentLab):
    def acquire(self, probe, channels=()):
        channels = self._selected(channels)
        history, observations = [], []
        previous_control = previous_reading = 0.
        for point in probe.sequence:
            control = point["u"]
            reading = abs(.8 * control + previous_control)
            target = .12 + .35 * control + reading ** 2 + .45 * previous_reading
            features = {"u": control, **({"meter": reading} if channels else {})}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
            previous_control, previous_reading = control, reading
        meter_cost = len(observations) * (3 if channels else 0)
        self.observation_cost += len(observations)
        self.measurement_cost += meter_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "channels": list(channels),
                                  "sequence": list(probe.sequence)})
        return MeasurementBatch(observations, len(observations), meter_cost, channels)


def test_unexplained_control_response_returns_to_real_meter_and_keeps_the_failed_direction(source):
    library = KnowledgeLibrary.from_dict(source[0])
    report = run_task(library, 2, laboratory=AbsoluteInstrument)
    comparisons = [action for action in report["actions"]
                   if action["action"] == "confirm_observation_research"
                   and action["research_action"] == "revise_mechanism"]
    assert comparisons and not any(action["accepted"] for action in comparisons)
    assert report["costs"]["observation_cost"] <= CONFIG.observation_budget
    assert report["costs"]["compute_cost"] <= CONFIG.compute_budget
    assert report["costs"]["measurement_cost"] > 0 and report["selected_channels"] == ["meter"]
    paid = next(index for index, action in enumerate(report["actions"])
                if action["action"] == "acquire" and action["measurement_cost"])
    assert all(report["actions"].index(action) < paid for action in comparisons)
    assert any(action["action"] == "instrument_mapping_counterexample" for action in report["actions"][paid:])


def test_polynomial_coordinates_preserve_startup_gates_and_do_not_reinvent_unsupported_operations():
    u = Program("feature", name="u")
    lag = Program("lag", name="u", lag=1)
    available = Program("available", lag=2)
    body = Program("add", children=(u, lag))
    program = Program("if", children=(available, body, Program("const", value=0.)))
    terms = _polynomial(program, SearchBudget(100))
    history = []
    for value in (.3, -.4, .8, -.9):
        row = Observation({"u": value}, 0., tuple(history))
        assert sum(coefficient * _term(key).predict(row) for key, coefficient in terms.items()) == program.predict(row)
        history.append({"u": value})
    with DynamicInstrumentLab() as lab:
        budget = SearchBudget(0)
        assert not RetainedControlRevisions([program], lab.spec, budget, max_history=4).coordinates
        assert budget.spent == 0
        nonlinear = Program("abs", children=(program,))
        assert not RetainedControlRevisions([nonlinear], lab.spec, SearchBudget(4000), max_history=4).coordinates
        recurrent = Program("accumulate", value=.48, children=(u, Program("const", value=0.)))
        retired = Program("add", children=(Program("mul", children=(Program("const", value=0.), recurrent)), program))
        active = Program("add", children=(Program("mul", children=(Program("const", value=1.), recurrent)), program))
        assert _polynomial(retired, SearchBudget(4000)) == terms
        assert RetainedControlRevisions([retired], lab.spec, SearchBudget(4000), max_history=4).coordinates
        assert _polynomial(active, SearchBudget(4000)) is None
        assert not RetainedControlRevisions([active], lab.spec, SearchBudget(4000), max_history=4).coordinates
