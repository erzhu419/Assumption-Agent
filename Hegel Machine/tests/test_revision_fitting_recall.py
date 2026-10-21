"""The ordinary PR2 route must pass the declared whole-context fitting budget."""

from copy import deepcopy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.theory_consolidation import _save
from test_question_routing import CONTEXT, future_question, probe, route, theory
from test_revision_context_fitting import FITTING, examples
from test_revision_relations import binary, constant, edit_candidate, feature
from test_theory_consolidation import agent_scene, assert_no_new_science


def test_automatic_frozen_recall_fits_the_open_hole_and_reaches_parent_experiment(tmp_path):
    agent, lab = agent_scene(tmp_path)
    frozen = {**edit_candidate(), 'version': 1}
    agent.theory_consolidation = {'candidates': {frozen['id']: frozen}}
    _save(agent)
    parent = binary('mul', constant(5.), feature())
    question = future_question(agent)
    question['theory_target'].update(parent_theory_id='parent',
        revision_fitting={**FITTING, 'operations': ['refit'], 'max_structures': 1},
        evidence_checks=[{**item, 'context_ref': deepcopy(CONTEXT)} for item in examples(parent)])

    result = route(agent, lab, question=question, theories=[theory('parent', parent)],
                   experiments=[probe(agent)])

    recall = result['frozen_recall']
    assert recall['generated_theory_ids'] and not recall['author_work']
    generated = [agent.theory_index[identifier] for identifier in recall['generated_theory_ids']]
    assert any(Program.from_dict(item['program']).children[0] == parent for item in generated)
    assert result['selection']['selected_experiment']
    assert result['parent_use']['status'] == 'parent_query_prepared'
    assert agent.theory_index['parent']['program'] == parent.to_dict()
    assert recall['new_compute_cost'] > 0
    assert_no_new_science(agent, lab)
