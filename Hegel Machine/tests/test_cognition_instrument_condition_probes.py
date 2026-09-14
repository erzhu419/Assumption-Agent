"""Shared instrument cuts become paid input questions without apparatus labels."""

from hegel_machine.cognition.instrument_condition_probes import instrument_condition_probe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def test_comparison_orientation_and_shared_fields_generate_only_unmeasured_control_cuts():
    axis = Program("feature", name="dial")
    below = Program("ge", children=(Program("const", value=.25), axis))
    above = Program("lt", children=(Program("const", value=.25), axis))
    far = Program("ge", children=(Program("const", value=2.75), axis))
    binding = {"a": below, "b": Program("and", children=(above, far)),
               "c": Program("not", children=(far,)), "value": Program("feature", name="drive")}
    spec = TaskSpec("inputs", "", ("dial", "drive"), (), (),
                    feature_bounds={"dial": (-2., 4.), "drive": (-1., 1.)})
    observed = [(Observation({"dial": .25, "drive": -.7}, 999.),
                 Observation({"a": 1., "b": 0., "c": 0., "value": -.7}, -999.))]
    budget = SearchBudget(200)
    probe, questions = instrument_condition_probe(binding, spec, [observed], budget,
        reference={"dial": 1., "drive": .8}, serial="current")
    assert probe.sequence == ({"dial": 2.75, "drive": .8},)
    assert questions == [{"control": "dial", "value": 2.75, "fields": ["b", "c"]}]
    assert 0 < budget.spent < budget.limit

    # Actual endpoint readings, regardless of response, settle this input
    # question. Repeating the failed law does not purchase this cut again.
    observed.append((Observation(probe.sequence[0], -100.),
                     Observation({"a": 0., "b": 1., "c": 0., "value": .8}, 100.)))
    assert instrument_condition_probe(binding, spec, [observed], budget,
        reference=probe.sequence[0], serial="later") == (None, [])


def test_numeric_maps_do_not_create_queries_and_exhausted_search_does_not_issue_a_partial_batch():
    spec = TaskSpec("inputs", "", ("dial",), (), (), feature_bounds={"dial": (-2., 4.)})
    axis = Program("feature", name="dial")
    numeric = Program("mul", children=(Program("const", value=.25), axis))
    assert instrument_condition_probe({"numeric": numeric}, spec, [], SearchBudget(20),
        reference={"dial": 0.}, serial="numeric") == (None, [])
    cut = Program("ge", children=(axis, Program("const", value=.25)))
    budget = SearchBudget(1)
    assert instrument_condition_probe({"a": cut}, spec, [], budget,
        reference={"dial": 0.}, serial="limited") == (None, [])
    assert budget.spent == budget.limit
