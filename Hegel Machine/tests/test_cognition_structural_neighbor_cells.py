"""A weak paid edge must remain useful after the average error meets its goal."""

import json
from pathlib import Path

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_programs import _Fitter, _entry
from hegel_machine.cognition.structural_discovery import discover_structure
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_structure import propose_structures


def paid_response():
    data = json.loads((Path(__file__).parent / 'fixtures' / 'relation_neighbor_paid_rows.json').read_text())
    schema, rows = ObjectSchema.from_dict(data['schema']), []
    for trajectory in data['trajectories']:
        history = []
        for values in trajectory:
            features = dict(zip(data['features'], values[:-1]))
            rows.append(Observation(features, values[-1], tuple(history)))
            history.append({**features, '__target__': values[-1]})
    response = [Program('object_sum', children=tuple(
        Program('feature', name=schema.fields[item][role]) for item in schema.objects))
        for role in ('q', 'u', 'x')]
    return rows, schema, response


def test_neighboring_paid_cell_recovers_the_weak_edge_without_more_observations():
    rows, schema, response = paid_response()
    work = SearchBudget(9000)
    found = discover_structure(rows, schema, budget=work, families=('joint_event',),
                               response_terms=response)
    assert len(rows) == 82 and 0 < work.spent <= work.limit
    assert found.fits[0]['training_loss'] < 1e-16
    assert found.fits[0]['effective_dimension'] == 3
    assert max((found.candidates[0].program.predict(row) - row.target) ** 2 for row in rows) < 1e-16
    # This rejected local fit previously became the entire retained domain:
    # its average error passed although two purchased rows contradicted it.
    weak = next((candidate, fit) for candidate, fit in zip(found.candidates, found.fits)
                if 1e-10 < fit['training_loss'] < 1e-7)
    assert max((weak[0].program.predict(row) - row.target) ** 2 for row in rows) > 1e-7
    assert found.radius_interval[1] <= weak[1]['radius_interval'][0]


def test_explicit_radius_question_does_not_expand_to_a_neighboring_domain():
    rows, schema, response = paid_response()
    radius = .9872102935595741  # The old paid fit, not an apparatus parameter.
    work = SearchBudget(2000)
    found = discover_structure(rows, schema, budget=work, families=('joint_event',),
                               response_terms=response, radius_candidates=[radius])
    assert found.fits and all(fit['radius'] == radius for fit in found.fits)
    assert 1e-10 < found.fits[0]['training_loss'] < 1e-7
    assert work.spent <= work.limit


def test_validation_winner_reopens_its_fitted_cell_when_research_consumes_all_paid_rows():
    rows, schema, response = paid_response()
    spec = TaskSpec('paid_response', '', schema.feature_names, (), (),
                    context={'object_schema': json.dumps(schema.to_dict())})
    work = SearchBudget(43234)
    fit = _Fitter(rows[:34], work, .5).fit(response)
    partial = {**_entry(fit, [], 34, work.spent), 'partial_object_knowledge': True}
    entries = propose_structures(spec, rows[:34], work, parent=partial, tolerance=1e-7)
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=1e-7))
    winner = agent._best([{**item, 'validation_loss': loss(item['candidate'].program,
                        rows[34:58], 0., .5)} for item in entries])
    assert winner['validation_loss'] < 1e-7 and winner['training_loss'] > 1e-10
    old = winner['structure']['radius_interval']
    continuation = SearchBudget(9000)
    revised = propose_structures(spec, rows, continuation, parent=winner, tolerance=1e-7)
    chosen = agent._best([{**item, 'validation_loss': item['training_loss']} for item in revised])
    assert chosen['training_loss'] < 1e-16
    assert chosen['structure']['radius_interval'][1] <= old[0]
    assert max((chosen['candidate'].program.predict(row) - row.target) ** 2 for row in rows) < 1e-16
    assert continuation.spent <= continuation.limit
