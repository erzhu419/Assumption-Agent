"""Removal changes actual Common proposals; fixture behavior is not science gain."""

from copy import deepcopy
import json

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.theory_consolidation import (
    _read_library, _save, install_theory_library,
)
from test_question_routing import CONTEXT, future_question, probe, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene, assert_no_new_science, consolidate


def test_keep_one_alias_and_whole_class_enter_the_same_real_common_route(tmp_path):
    source, laboratory = agent_scene(tmp_path / 'source')
    consolidate(source, laboratory)
    old = deepcopy(source.theory_consolidation['candidates'])
    learned = {**edit_candidate(), 'version': 1}
    additions = {name: {**deepcopy(learned), 'id': name} for name in ('new_a', 'new_b')}
    library = source.workspace / 'frozen_library.json'
    library.write_text(json.dumps({'candidates': {**old, **additions}}))
    reference = {'file': str(library), 'json_path': [], 'workspace': str(source.workspace)}
    original = _read_library(reference, tmp_path)
    parent = binary('mul', constant(5.), feature())
    outcomes = {}
    for name, removed in (('keep', []), ('minus_one', ['new_a']),
                          ('minus_class', ['new_a', 'new_b'])):
        view = tmp_path / f'{name}.json'
        view.write_text(json.dumps({'source_ref': reference,
                                    'removed_candidates': removed}))
        agent, lab = agent_scene(tmp_path / name)
        agent.theory_consolidation = install_theory_library(agent,
            {'file': str(view), 'json_path': []})
        _save(agent)
        installed = _read_library({'file': str(view), 'json_path': []}, tmp_path)
        assert all(installed['candidates'][key] == original['candidates'][key] for key in old)
        question = future_question(agent)
        question['theory_target'].update(parent_theory_id='parent',
            revision_fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1},
            evidence_checks=[{**check, 'context_ref': deepcopy(CONTEXT)}
                             for check in examples(parent)])
        result = agent._execute('run', {'operator': 'research_route',
            'research_question': question, 'theories': [theory('parent', parent)],
            'experiments': [probe(agent)], 'allocation': {'compute': 6000}}, lab)
        programs = {json.dumps(Program.from_dict(entry['program']).expanded().to_dict(),
                               sort_keys=True)
                    for entry in agent.theory_index.values() if entry.get('program')}
        forecasts = {round(Program.from_dict(json.loads(program)).predict(
            Observation({'u': 3.}, 0.)), 7) for program in programs}
        outcomes[name] = {'programs': programs, 'forecasts': forecasts,
            'used': {work['candidate_id'] for work in result['frozen_recall']['work']
                     if work['theory_ids']}}
        assert agent.theory_index['parent']['program'] == parent.to_dict()
        assert result['frozen_recall']['new_compute_cost'] <= 6000
        assert not result['frozen_recall']['support_updated']
        assert_no_new_science(agent, lab)
    assert {'new_a', 'new_b'} <= outcomes['keep']['used']
    assert 'new_a' not in outcomes['minus_one']['used']
    assert 'new_b' in outcomes['minus_one']['used']
    # An alias ID disappearing is not a different executable or forecast.
    assert outcomes['keep']['programs'] == outcomes['minus_one']['programs']
    assert outcomes['keep']['forecasts'] == outcomes['minus_one']['forecasts']
    assert not {'new_a', 'new_b'} & outcomes['minus_class']['used']
    assert outcomes['keep']['programs'] != outcomes['minus_class']['programs']
    assert 22. in outcomes['keep']['forecasts']
    assert 22. not in outcomes['minus_class']['forecasts']
