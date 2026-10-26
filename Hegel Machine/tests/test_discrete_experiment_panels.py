"""Public discrete commands must be legal before any observation is bought."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.conditional_experiments import (
    propose_conditional_boundary_probe, propose_conditional_experiments,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget, TaskSpec
from hegel_machine.cognition.program_disagreement_probes import (
    _forecast, propose_program_disagreement_probes,
)
from hegel_machine.cognition.scale_bindings import bind_measurement_arguments


def public_commands():
    lab = AnonymousHistoryLab()
    public = lab.public_spec()
    spec = TaskSpec("public_discrete", "Use the public command contract.",
        tuple(public["controls"]), (), (), feature_bounds=public["feature_bounds"],
        max_probe_length=public["max_probe_length"])
    return lab, public, spec


def threshold(value, name="u"):
    return Program("ge", children=(Program("feature", name=name), Program("const", value=value)))


def assert_public_probe(lab, public, probe):
    action = {"kind": "trajectory", "channels": [], "sequence": list(probe.sequence)}
    rows = lab.prediction_rows(action)
    assert lab.action_cost(action) == len(rows) == len(probe.sequence)
    assert 0 < len(rows) <= public["max_probe_length"]
    assert all(point["u"] in public["control_levels"]["u"] for point in probe.sequence)
    return rows


def test_program_panel_reforecasts_legal_levels_without_observing_the_apparatus():
    lab, public, spec = public_commands()
    programs = [threshold(.21), threshold(.39)]
    work = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, work,
        length=48, control_levels=public["control_levels"])
    assert proposals
    for proposal in proposals:
        assert_public_probe(lab, public, proposal["probe"])
        forecasts = [_forecast(program, proposal["probe"].sequence, lambda units: True)
                     for program in programs]
        assert proposal["predictions"] == forecasts
        assert proposal["predicted_separation"] > 0.
        assert any(left != right for left, right in zip(*forecasts))
    assert 0 < work.spent <= work.limit
    assert lab.costs == {"observation_cost": 0, "measurement_cost": 0, "queries": 0}


def test_continuous_disagreement_that_disappears_on_legal_levels_is_not_returned():
    lab, public, spec = public_commands()
    programs = [threshold(.318100001), threshold(.318100002)]
    assert propose_program_disagreement_probes(spec, programs, SearchBudget(6000))
    work = SearchBudget(6000)
    assert propose_program_disagreement_probes(spec, programs, work,
        control_levels=public["control_levels"]) == []
    assert 0 < work.spent <= work.limit
    assert lab.costs["queries"] == 0


def test_temporal_binding_uses_the_same_public_levels_and_actual_forecasts():
    lab, public, spec = public_commands()
    programs = [threshold(.21, "reading"), threshold(.39, "reading")]
    previous = Program("if", children=(Program("available", lag=2),
        Program("lag", name="u", lag=2), Program("const", value=0.)))
    binding = {"reading": previous}
    work = SearchBudget(6000)
    proposals = propose_program_disagreement_probes(spec, programs, work,
        binding=binding, length=24, control_levels=public["control_levels"])
    assert proposals
    executable = [bind_measurement_arguments(program, binding) for program in programs]
    for proposal in proposals:
        assert proposal["reason"] == "temporal_program_predicate"
        assert_public_probe(lab, public, proposal["probe"])
        assert proposal["predictions"] == [_forecast(program, proposal["probe"].sequence,
            lambda units: True) for program in executable]
        assert proposal["predicted_separation"] > 0.
    assert 0 < work.spent <= work.limit
    assert lab.costs["queries"] == 0


def test_conditional_packs_keep_matched_prefixes_and_quote_actual_legal_rows():
    lab, public, spec = public_commands()
    work = SearchBudget(6000)
    packs = propose_conditional_experiments(spec, [], [threshold(.21), threshold(.39)], work,
        observation_budget=48, control_levels=public["control_levels"])
    assert len(packs) == 2
    assert packs[0]["background"] != packs[1]["background"]
    for pack in packs:
        probes = [pack["reference"], *(trial["probe"] for trial in pack["interventions"])]
        assert pack["observation_cost"] == sum(len(probe.sequence) for probe in probes)
        for probe in probes:
            assert_public_probe(lab, public, probe)
        for trial in pack["interventions"]:
            prefix = trial["prefix_length"]
            assert trial["probe"].sequence[:prefix] == pack["reference"].sequence[:prefix]
            assert all(level in public["control_levels"]["u"] for level in trial["levels"])
    assert 0 < work.spent <= work.limit
    assert lab.costs["queries"] == 0


def test_conditional_boundary_uses_only_available_levels_inside_measured_interval():
    lab, public, spec = public_commands()
    evidence = {"control": "u", "background": {"u": .31}, "threshold_interval": [.13, .47]}
    probe = propose_conditional_boundary_probe(spec, evidence, length=48,
        control_levels=public["control_levels"])
    assert probe is not None
    assert_public_probe(lab, public, probe)
    assert all(.13 <= point["u"] <= .47 for point in probe.sequence[1:])
    original = deepcopy(evidence)
    evidence["threshold_interval"] = [.318100001, .318100002]
    assert propose_conditional_boundary_probe(spec, evidence,
        control_levels=public["control_levels"]) is None
    assert original["background"] == evidence["background"] == {"u": .31}
    assert lab.costs["queries"] == 0


def test_a_missing_discrete_control_contract_is_not_filled_from_bounds():
    _, _, spec = public_commands()
    with pytest.raises(ValueError, match="control_levels.*u"):
        propose_program_disagreement_probes(spec, [threshold(.21), threshold(.39)],
            SearchBudget(6000), control_levels={})


def test_legalization_is_charged_before_forecasting_and_stops_with_the_budget():
    _, public, spec = public_commands()
    work = SearchBudget(10)
    assert propose_program_disagreement_probes(spec, [threshold(.21), threshold(.39)], work,
        control_levels=public["control_levels"]) == []
    assert 0 < work.spent <= work.limit
    work = SearchBudget(20)
    assert propose_conditional_experiments(spec, [], [], work,
        control_levels=public["control_levels"]) == []
    assert 0 < work.spent <= work.limit
