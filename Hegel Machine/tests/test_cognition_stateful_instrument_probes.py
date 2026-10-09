"""Recurrent observation models determine executable multi-step experiments."""

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import propose_program_disagreement_probes
from hegel_machine.cognition.temporal_control_probes import _Prefix


def state(decay, initial=0.):
    return Program("accumulate", value=decay, children=(Program("feature", name="u"),
                                                       Program("const", value=initial)))


def test_prefix_unroll_preserves_actual_recurrent_states_and_temporal_readouts():
    budget = SearchBudget(6000)
    prefix = _Prefix({"u": (-1., 1.)}, 4, budget.spend)
    mixed = Program("add", children=(state(.73, .21), state(-.32, -.4)))
    program = Program("add", children=(Program("window_sum", lag=3, children=(mixed,)),
        Program("event_delta", lag=2, children=(mixed,))))
    sequence = ({"u": .7}, {"u": -.3}, {"u": .1}, {"u": .8}, {"u": -.6})
    symbolic = prefix.program(program)
    flat = {alias: sequence[step][name] for (name, step), alias in prefix.coordinates.items()}
    assert symbolic.predict(Observation(flat, 0.)) == pytest.approx(
        program.predict(Observation(sequence[-1], 0., sequence[:-1])))
    assert budget.spent <= budget.limit


@pytest.mark.parametrize("decay", (.65, .82))
def test_discriminating_experiment_builds_state_beyond_single_step_reach(decay):
    meter = Program("feature", name="meter")
    gate = Program("and", children=(
        Program("ge", children=(meter, Program("const", value=2.2))),
        Program("lt", children=(meter, Program("const", value=2.20001)))))
    claims = [Program("if", children=(gate, Program("const", value=gain),
                                       Program("const", value=0.))) for gain in (1., 2.)]
    sensor = state(decay)
    spec = TaskSpec("recurrent_gate", "Apply public controls.", ("u",), (), (),
                    feature_bounds={"u": (-1., 1.)}, max_probe_length=12)
    budget = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, claims, budget, binding={"meter": sensor})
    assert proposals and budget.spent <= budget.limit
    selected = proposals[0]
    assert selected["boundary"]["solver_strategy"] == "temporal_prefix"
    history, readings = [], []
    for point in selected["probe"].sequence:
        readings.append(sensor.predict(Observation(point, 0., tuple(history))))
        history.append(point)
    assert any(2.2 <= reading < 2.20001 for reading in readings)
    assert all(abs(reading) < 2.2 for reading in readings[:2])
    for claim, predictions in zip(claims, selected["predictions"]):
        assert predictions == pytest.approx([claim.predict(Observation({"meter": value}, 0.))
                                            for value in readings])
