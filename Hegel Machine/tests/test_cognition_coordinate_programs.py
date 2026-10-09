"""Current observations can start arithmetic research without prior concepts."""

import json
from random import Random

import pytest

from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import (
    propose_coordinate_programs, replay_research_program,
)


def evidence(seed, response, count=48):
    rng = Random(seed)
    return [Observation(point, response(**point)) for point in
            ({'drive': rng.uniform(-1.4, 1.4), 'load': rng.uniform(-1.2, 1.2)}
             for _ in range(count))]


SPEC = TaskSpec('measured_apparatus', 'Predict measured response', ('drive', 'load'), (), ())


@pytest.mark.parametrize('response', [
    lambda drive, load: .2 + 1.1 * (.8 * drive ** 2 - .65 * load) ** 2,
    lambda drive, load: -.3 + (.9 * drive ** 2 + 1.2 * load ** 2) ** 2,
    lambda drive, load: .7 * drive ** 3 - 1.4 * drive * load + .1,
    lambda drive, load: 1.3 * drive ** 4 + .25,
])
def test_unseen_response_is_discovered_without_source_calls(response):
    budget = SearchBudget(32000)
    claims = propose_coordinate_programs(SPEC, evidence(19, response), budget)
    assert claims
    selected = min(claims, key=lambda item: item['training_loss'])
    program = selected['candidate'].program
    assert sum((program.predict(row) - row.target) ** 2
               for row in evidence(91, response, 64)) / 64 < 1e-7
    assert not selected['candidate'].parents
    assert selected['research_program']['status'] == 'requires_fresh_validation'
    assert any(step['operation'] == 'compose_measured_coordinates'
               for step in selected['research_program']['selection_steps'])
    assert 0 < budget.spent <= budget.limit


def test_discovered_coordinates_survive_json_and_refit_after_sensor_rename():
    response = lambda drive, load: .2 + (.8 * drive ** 2 - .65 * load) ** 2
    claims = propose_coordinate_programs(SPEC, evidence(19, response), SearchBudget(32000))
    selected = min(claims, key=lambda item: item['training_loss'])
    plan = json.loads(json.dumps(selected['research_program']))
    changed = lambda drive, load: -.4 + 1.7 * (.8 * drive ** 2 + .95 * load) ** 2

    def renamed(seed):
        return [Observation({'u': row.features['drive'], 'v': row.features['load']}, row.target)
                for row in evidence(seed, changed)]

    budget = SearchBudget(2000)
    replayed = replay_research_program(plan, renamed(201), budget,
                                       feature_mapping={'drive': 'u', 'load': 'v'})
    assert replayed is not None
    assert sum((replayed['candidate'].program.predict(row) - row.target) ** 2
               for row in renamed(311)) / 48 < 1e-7
    assert replayed['research_program']['reused_procedure']
    assert 0 < budget.spent <= budget.limit


def test_unfunded_coordinate_search_produces_no_candidate():
    budget = SearchBudget(2)
    assert propose_coordinate_programs(SPEC, evidence(5, lambda drive, load: drive ** 4), budget) == []
    assert budget.spent == 0
