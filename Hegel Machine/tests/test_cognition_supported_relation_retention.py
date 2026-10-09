"""Paid evidence can narrow uncertainty without replacing its supported answer."""
from copy import deepcopy
import json

from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_retained_relation_intervals import composition, observations, prior, mse


def test_existing_radius_survives_refinement_and_json_without_being_a_sampled_midpoint():
    parent = composition(.6)
    record = deepcopy(parent.to_dict())
    paid = observations(parent, (.5999998, .6000006, 1.2, .1) * 3)
    research = RelationIntervalResearch(parent, [prior((.5, .8))], SearchBudget(10000))
    entries = research.alternatives(paid, SearchBudget(50000))
    assert entries[0]['parameter_values'] == research.original
    assert entries[0]['candidate'].program == parent
    assert mse(entries[0]['candidate'].program, observations(parent, (.6, 1.2, .6, 1.2))) == 0.
    saved = json.loads(json.dumps(entries[0]['research_program']))
    restored = RelationIntervalResearch(parent, [], SearchBudget(10000), research_program=saved)
    assert restored.values == research.original
    assert restored.intervals[0]['radius_interval'][1] - restored.intervals[0]['radius_interval'][0] <= 1e-6
    assert parent.to_dict() == record


def test_real_counterexample_still_removes_the_incumbent():
    parent, truth = composition(.6), composition(.72)
    paid = observations(truth, (.1, .61, .68, 1.2) * 4)
    research = RelationIntervalResearch(parent, [prior((.5, .8))], SearchBudget(10000))
    entries = research.alternatives(paid, SearchBudget(50000), experiments=[paid])
    assert entries and mse(parent, paid) > 1e-7
    assert all(entry['candidate'].program != parent for entry in entries)
    assert all(mse(entry['candidate'].program, paid) < 1e-7 for entry in entries)
    assert all(entry['parameter_values'] != research.original for entry in entries)
