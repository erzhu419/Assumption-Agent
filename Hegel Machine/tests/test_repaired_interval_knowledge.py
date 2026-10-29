"""A method-carrier repair changes no confirmed answer or old knowledge."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from scripts import successor_interval_scene as scene_io
from scripts.run_partial_role_parent import save
from test_relation_fitted_columns import fitted_entry, interval_answer, measured_rows


def knowledge_fixture(tmp_path, monkeypatch, *, prior_inputs=True):
    repaired = interval_answer(fitted_entry(), .7)
    rows = measured_rows(repaired['candidate'].program)
    fields = {'a': {'f': 'f0', 'x': 'x0', 'q': 'q0',
                    **{f'z{i}': f'z{i}' for i in range(7)}},
              'b': {'f': 'f1', 'x': 'x1', 'q': 'q1'}}
    schema = {'objects': ['a', 'b'], 'fields': fields,
              'members': {'a': ['a'], 'b': ['b']}, 'controls': deepcopy(fields)}
    spec = TaskSpec('measured_source', '', tuple(rows[0].features), (), (),
                    context={'object_schema': json.dumps(schema)})
    agent = UnifiedResearchAgent(library=KnowledgeLibrary())
    concept, _ = agent.library.learn(repaired['candidate'], spec, rows,
                                      paid_support_matches=[True]*len(rows))
    collapsed = _plan({'program': repaired['candidate'].program,
        'fit_terms': [repaired['candidate'].program], 'coefficients': [0., 1.], 'loss': 0.},
        [], len(rows), 0)
    collapsed.update(status='supported_on_fresh_trajectory', confirmation_loss=0.,
                     confirmation_rows=48, confirmation_ref={'file': 'original_paid_confirmation.json'})
    agent.library.meta_priors.extend([{'source_concept_id': 'earlier', 'notes': 'retain old failure'},
        {'source_concept_id': concept, 'status': 'supported',
         'research_program': collapsed, 'input_research_program': deepcopy(collapsed),
         'counterexamples': [{'task': 'old_negative'}]}])
    if not prior_inputs:
        agent.library.meta_priors[-1].pop('input_research_program')
    agent.library.program_experiment_methods.records = [{'method_id': str(i)} for i in range(5)]
    before = deepcopy(agent.library.to_dict())
    root = tmp_path / 'K2'
    selected_ref = scene_io.ref(root / 'selected.json')
    repaired_ref = scene_io.ref(tmp_path / 'repair.json')
    save(root / 'selected.json', encode_entry(repaired))
    save(root / 'selection.json', {'selected_ref': selected_ref})
    save(root / 'receipt.json', {'knowledge': {'concept_id': concept}, 'cumulative_compute': 269243})
    save(root / 'knowledge_delta.json', {'source': str(tmp_path / 'source')})
    save(tmp_path / 'repair.json', encode_entry(repaired))
    monkeypatch.setattr(scene_io, 'load_confirmed_knowledge',
                        lambda path: KnowledgeLibrary.from_dict(deepcopy(before)))
    monkeypatch.setattr(scene_io, 'restore_interval_scene',
                        lambda path: {'agent': agent, 'spec': spec})
    return root, repaired_ref, before


@pytest.mark.parametrize('prior_inputs', [True, False])
def test_repaired_method_uses_normal_measured_input_save_without_world_changes(tmp_path, monkeypatch, prior_inputs):
    root, repaired_ref, before = knowledge_fixture(tmp_path, monkeypatch, prior_inputs=prior_inputs)
    files = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    monkeypatch.setattr('hegel_machine.cognition.language.Program.predict',
        lambda *a, **kw: pytest.fail('Knowledge repair must not predict'))
    monkeypatch.setattr('hegel_machine.cognition.research_programs._Fitter.fit',
        lambda *a, **kw: pytest.fail('Knowledge repair must not refit'))
    library, receipt = scene_io.load_repaired_knowledge(root, repaired_ref)
    after = library.to_dict()
    assert receipt['prior_index'] == 1 and receipt['world_program_unchanged']
    assert receipt['new_fits'] == receipt['new_predictions'] == receipt['new_observations'] == 0
    assert receipt['actions'][-1]['input_research_program_preserved']
    prior = after['meta_priors'][1]
    assert len(prior['research_program']['term_nodes']) == 8
    assert len(prior['input_research_program']['term_nodes']) == 8
    assert len(prior['input_research_program']['selection_steps'][-1]['parameter_terms']) == 8
    assert set(prior['input_research_program']['input_features']) == {
        'f0', 'x0', 'q0', 'f1', 'x1', 'q1', *(f'z{i}' for i in range(7))}
    assert 'z6' in prior['input_research_program']['input_template']['source_features']
    assert prior['research_program']['confirmation_ref'] == {'file': 'original_paid_confirmation.json'}
    assert prior['research_program']['status'] == 'supported_on_fresh_trajectory'
    expected = deepcopy(before)
    for name in ('research_program', 'input_research_program'):
        expected['meta_priors'][1][name] = prior[name]
    assert after == expected
    assert {p: p.read_bytes() for p in tmp_path.rglob('*.json')} == files


def test_different_world_answer_cannot_be_installed_as_a_carrier_repair(tmp_path, monkeypatch):
    root, repaired_ref, before = knowledge_fixture(tmp_path, monkeypatch)
    other = interval_answer(fitted_entry(), .6)
    save(tmp_path / 'repair.json', encode_entry(other))
    monkeypatch.setattr(scene_io, 'load_confirmed_knowledge',
                        lambda *a: pytest.fail('Different world answer must stop before editing knowledge'))
    with pytest.raises(ValueError, match='changes the frozen selected world answer'):
        scene_io.load_repaired_knowledge(root, repaired_ref)
