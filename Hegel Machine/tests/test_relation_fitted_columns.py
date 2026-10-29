"""Interval answers retain the fitted coordinates needed by later research."""

from copy import deepcopy
from dataclasses import replace
import json
import random

from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch, _bind
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.research_programs import _plan, replay_research_program
from hegel_machine.cognition.structural_research_program import attach_structural_research_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_retained_relation_evidence import consumer_call, saved_source


def coordinates(radius):
    neighbor = Program('neighbor_sum', children=(Program('const', value=radius),
        *(Program('feature', name=name) for name in ('f0', 'x0', 'q0', 'f1', 'x1', 'q1'))))
    response = Program('window_sum', lag=3, children=(neighbor,))
    return [response, *(Program('feature', name=f'z{i}') for i in range(7))]


SOURCE_WEIGHTS = [1.1, .7, -.4, .2, -.6, .8, .3, 0.]


def fitted_entry():
    terms = coordinates(.8)
    program = _sum_terms(terms, SOURCE_WEIGHTS, .25)
    entry = {'candidate': Candidate('saved_structure', program, 'representation'),
        'kind': 'structural:joint_window_event', 'family_id': 'T12', 'training_loss': 0.,
        'validation_loss': 0., 'structure': {'compute_cost': 100,
            'radius_interval': [.4, 1.], 'fits': [{'family': 'joint_window_event',
                'window': 3, 'radius': .8, 'radius_interval': [.4, 1.],
                'fit_terms': [term.to_dict() for term in terms],
                'coefficients': [.25, *SOURCE_WEIGHTS]}]}}
    return attach_structural_research_program(entry, fit_rows=64)


def interval_answer(entry, radius, *, with_structure=True):
    research = RelationIntervalResearch(entry['candidate'].program, [], SearchBudget(100000),
        structure=entry.get('structure') if with_structure else None,
        research_program=entry.get('research_program'))
    assert len(research.parameters) == 1
    values = {research.parameters[0]['symbol']: radius}
    return research._entry(values, [.4, 1.], 0, 0., 64, 0)


def graph_terms(entry):
    plan = entry['research_program']
    graph = ResearchGraph.from_dict(plan['graph'])
    return [graph.nodes[key].answer for key in plan['term_nodes']]


def parameter_step(entry):
    return next(step for step in reversed(entry['research_program']['selection_steps'])
                if step['operation'] == 'resolve_relation_intervals')


def measured_rows(program):
    rng = random.Random(913)
    result = []
    for trajectory in range(12):
        history = []
        for index in range(8):
            features = {'f0': 1., 'f1': 0., 'x0': 0., 'x1': rng.uniform(.2, 1.2),
                'q0': rng.uniform(-1., 1.), 'q1': rng.uniform(-1., 1.),
                **{f'z{i}': rng.uniform(-1., 1.) for i in range(7)}}
            row = Observation(features, 0., tuple(history))
            row = replace(row, target=program.predict(row))
            result.append(row)
            history.append({**features, '__target__': row.target})
    return result


def test_interval_answer_preserves_eight_columns_and_parameter_bindings_without_refit(monkeypatch):
    source = fitted_entry()
    frozen = deepcopy(source)

    def no_refit(*args, **kwargs):
        raise AssertionError('Saving the completed interval answer must not fit again')

    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit', no_refit)
    unchanged = interval_answer(source, .8)
    changed = interval_answer(source, .7)
    assert source == frozen
    assert unchanged['candidate'].program == source['candidate'].program
    assert graph_terms(changed) == coordinates(.7)
    expected = _sum_terms(coordinates(.7), SOURCE_WEIGHTS, .25)
    assert changed['candidate'].program.expanded() == expected.expanded()
    saved = parameter_step(changed)
    bindings = {item['symbol']: Program('const', value=item['radius'])
                for item in saved['parameters']}
    assert len(saved['parameter_terms']) == 8
    assert [_bind(Program.from_dict(term), bindings).simplified()
            for term in saved['parameter_terms']] == coordinates(.7)
    assert _bind(Program.from_dict(saved['parameter_template']), bindings).simplified().expanded() == expected.expanded()


def test_serialized_interval_keeps_columns_for_next_update_and_new_task_weights():
    answer = interval_answer(fitted_entry(), .7)
    restored = decode_entry(json.loads(json.dumps(encode_entry(answer))))
    updated = interval_answer(restored, .65, with_structure=False)
    assert graph_terms(updated) == coordinates(.65)
    # Every coordinate is supported by the saved source fit. The final one was
    # fitted with zero source weight, but is needed by this new task's evidence.
    target = _sum_terms(coordinates(.65), [-.3, .1, .8, -.5, .9, -.2, .4, 1.7], -.6)
    observations = measured_rows(target)
    original = deepcopy(updated)
    replay = replay_research_program(updated['research_program'], observations, SearchBudget(5000))
    assert updated == original
    assert replay is not None and replay['research_program']['reused_procedure']
    assert len(graph_terms(replay)) == 8
    assert loss(replay['candidate'].program, observations, 0., .5) < 1e-12
    assert replay['candidate'].program != updated['candidate'].program
    saved = parameter_step(replay)
    assert len(saved['parameter_terms']) == 8
    bindings = {item['symbol']: Program('const', value=item['radius'])
                for item in saved['parameters']}
    assert _bind(Program.from_dict(saved['parameter_template']), bindings).simplified().expanded() == replay['candidate'].program.expanded()
    reopened = RelationIntervalResearch(replay['candidate'].program, [], SearchBudget(100000),
                                       research_program=replay['research_program'])
    assert len(reopened.parameters) == 1
    assert reopened.values[reopened.parameters[0]['symbol']] == .65


def test_old_complete_response_without_fitted_columns_keeps_its_original_path():
    response = coordinates(.8)[0]
    research = RelationIntervalResearch(response, [], SearchBudget(100000),
        structure={'fits': [{'radius': .8}], 'radius_interval': [.4, 1.]})
    answer = research._entry({research.parameters[0]['symbol']: .7}, [.4, 1.], 0, 0., 64, 0)
    assert graph_terms(answer) == [answer['candidate'].program]
    assert len(parameter_step(answer)['parameter_terms']) == 1
    assert answer['candidate'].program.expanded() == coordinates(.7)[0].expanded()


def test_equal_source_radii_in_distinct_coordinates_keep_separate_column_bindings():
    _, _, selected, prior = saved_source()
    terms = [consumer_call(selected, prefix=prefix, outer=False) for prefix in ('left_', 'right_')]
    parent = _sum_terms(terms, [1., -.6], .25)
    fit = {'program': parent, 'fit_terms': terms, 'coefficients': [.25, 1., -.6], 'loss': 0.}
    plan = _plan(fit, [], 64, 0)
    budget = SearchBudget(50000)
    research = RelationIntervalResearch(parent, [prior], budget, research_program=plan)
    assert len(research.parameters) == 2
    first, second = [item['symbol'] for item in research.parameters]
    assert research.values[first] == research.values[second]
    initial = deepcopy(research.values)
    changed = research._entry({**initial, first: .65}, [.4, 1.], 0, 0., 64, 0)
    changed_terms = graph_terms(changed)
    assert len(changed_terms) == 2
    assert changed_terms[0].expanded() != terms[0].expanded()
    assert changed_terms[1].expanded() == terms[1].expanded()

    restored = decode_entry(json.loads(json.dumps(encode_entry(changed))))
    again = RelationIntervalResearch(restored['candidate'].program, [], budget,
                                     research_program=restored['research_program'])
    assert len(again.parameters) == 2
    second_change = again._entry({**again.values, second: .55}, [.4, 1.], 1, 0., 64, 0)
    final_terms = graph_terms(second_change)
    assert final_terms[0].expanded() == changed_terms[0].expanded()
    assert final_terms[1].expanded() != changed_terms[1].expanded()
    assert budget.spent <= budget.limit


def test_normal_library_update_and_recall_keep_all_columns_for_successor_refit():
    answer = interval_answer(fitted_entry(), .7)
    source_rows = measured_rows(answer['candidate'].program)
    names = tuple(sorted(source_rows[0].features))
    source_spec = TaskSpec('source_columns', '', names, (), ())
    author = UnifiedResearchAgent(library=KnowledgeLibrary())
    concept, added = author.library.learn(answer['candidate'], source_spec, source_rows)
    assert added
    prior = {'source_concept_id': concept, 'status': 'supported',
             'source_features': list(names), 'research_program': fitted_entry()['research_program']}
    author.library.meta_priors.append(prior)
    actions = []
    assert author._remember_relation_research(prior, answer, source_spec, actions)
    assert actions[-1]['action'] == 'preserve_relation_research'
    frozen = deepcopy(author.library.to_dict())
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(frozen)))
    assert len(library.meta_priors[0]['research_program']['term_nodes']) == 8

    target = _sum_terms(coordinates(.7), [-.3, .1, .8, -.5, .9, -.2, .4, 1.7], -.6)
    target_rows = measured_rows(target)
    target_spec = TaskSpec('successor_columns', '', names, (), ())
    successor = UnifiedResearchAgent(library=library)
    calls = successor._recall(target_spec)
    plans = successor._replay_program_queue(target_spec, calls)
    assert plans and plans[0][0] == concept
    assert len(plans[0][1]['term_nodes']) == 8
    claims = successor._replay_program_claims(target_spec, target_rows, SearchBudget(10000), calls,
                                            prepared=plans)
    assert claims and all(claim['method_source_ids'] == [concept] for claim in claims)
    assert min(loss(claim['candidate'].program, target_rows, 0., .5) for claim in claims) < 1e-12
    assert author.library.to_dict() == frozen
