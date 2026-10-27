"""Registered Common results reuse workspace storage without weakening their archive."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import anonymous_research_archive as archives
from hegel_machine.cognition.anonymous_research_archive import (
    load_research_archive, resolve_workspace_archive,
)
from hegel_machine.cognition.code_agent import (
    archived_paid_trace_references, evaluate_python_model, retain_evaluated_python_model,
)
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.theory_consolidation import INDEX_PATH, LIBRARY_PATH
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from scripts.run_online_inquiry_llm import write_json
from test_native_condition_inquiry import PaidConditionalLab, buy_fixture, execute
from test_native_theory_registration import CONTEXT, TARGET
from test_paid_trace_references import MODEL, TRACE


def agent_at(root, knowledge, *, references=None, group='archive'):
    initial = root.parent / 'initial_theory_library.json'
    initial.parent.mkdir(parents=True, exist_ok=True)
    initial.write_text('{"candidates": {}}')
    return RegisteredOrganizationAgent(None, root, arm=group,
        research_organization={'phase': 'source', 'initial_library_path': str(initial),
            'theories': [], 'consolidation': {'source_theory_ids': [],
                'mission': {'purpose': 'theory_learning', 'mission_id': 'storage-fixture',
                            'theory_learning_allocation': {'compute': 100}}}},
        knowledge=knowledge, source_task=False, paid_trace_references=references,
        config=InquiryLLMConfig(max_model_calls=0, observation_budget=300,
            confirmation_rows=16, compute_budget=2_000_000, execution_seconds=30.))


def allocated(root):
    return sum(path.lstat().st_blocks * 512 for path in root.rglob('*')
               if not path.is_symlink())


def save_source(root, agent, result):
    write_json(root / 'protocol.json', {'cases': [{'id': 'case'}], 'arms': ['archive']})
    write_json(agent.workspace / 'training.json', result)
    write_json(agent.workspace / 'archive.json', result['outgoing_knowledge'])


@pytest.mark.parametrize(('declaration', 'saved_name'), [
    ('reference', 'training.json'), ('missing', 'research_state.json'), ('empty', 'training.json')])
def test_workspace_archive_keeps_bank_references_without_loading_bank_payloads(
        tmp_path, monkeypatch, declaration, saved_name):
    workspace = tmp_path / 'source'
    workspace.mkdir()
    model = 'def predict(row): return 0.\n'
    (workspace / 'model.py').write_text(model)
    references = {field: {'file': str(workspace / name), 'json_path': []}
                  for field, name in (('theory_index', INDEX_PATH),
                                      ('theory_consolidation', LIBRARY_PATH))}
    for reference in references.values():
        write_json(Path(reference['file']), {'retained': 'x' * 65536})
    write_json(workspace / saved_name, {'inherited_files': ['model.py'], 'events': [],
        'model': 'model.py', 'evaluated_model': None, 'theory_index': {'large_bank': 'x' * 65536}})
    declarations = (references if declaration == 'reference' else
                    {field: {} for field in references} if declaration == 'empty' else {})
    archive = {'workspace': str(workspace), 'lessons': ['Retain failed source evidence.'],
        'question_traces': [{'id': 'failed', 'status': 'refuted'}],
        **declarations}
    original = deepcopy(archive)
    read, selected_read = Path.read_text, archives.read_json_checkpoint
    selected = []

    def no_bank_text(path, *args, **kwargs):
        assert path.name != saved_name
        if declaration != 'empty':
            assert path.name not in {Path(INDEX_PATH).name, Path(LIBRARY_PATH).name}
        return read(path, *args, **kwargs)

    def projected(path, *, keys=None):
        assert keys == {'inherited_files', 'events', 'model', 'evaluated_model'}
        selected.append(Path(path))
        return selected_read(path, keys=keys)

    monkeypatch.setattr(Path, 'read_text', no_bank_text)
    monkeypatch.setattr(archives, 'read_json_checkpoint', projected)
    resolved = resolve_workspace_archive(archive)
    assert selected == [workspace / saved_name]
    assert archive == original
    expected = declarations if declaration == 'empty' else references
    assert all(resolved[field] == reference for field, reference in expected.items())
    assert resolved['files']['model.py'] == model
    if declaration == 'empty':
        for name in (INDEX_PATH, LIBRARY_PATH):
            assert json.loads(resolved['files'][name]) == {'retained': 'x' * 65536}
    else:
        assert INDEX_PATH not in resolved['files'] and LIBRARY_PATH not in resolved['files']
    assert resolved['lessons'] == archive['lessons']
    assert resolved['question_traces'] == archive['question_traces']


def test_registered_results_reference_inherited_trace_and_keep_writable_models_independent(tmp_path):
    trace = json.dumps({'gain': 3, 'events': ['retained paid fixture'] * 30_000})
    knowledge = {'files': {'model.py': MODEL, 'parameter.json': '{"bias": 5}', TRACE: trace}}
    write_json(tmp_path / 'initial_archive.json', knowledge)
    references = archived_paid_trace_references(knowledge['files'], tmp_path / 'reference_archive')
    fixed_allocation = allocated(tmp_path)
    agents, snapshots = [], []
    for group in ('archive', 'fixed_pr2'):
        agent = agent_at(tmp_path / group, knowledge, references=references, group=group)
        agents.append(agent)
        agent._prepare_program_state()
        result = agent.result('researching')
        assert result['knowledge_protocol'] == 'method_practice_workspace'
        assert result['outgoing_knowledge']['workspace'] == str(agent.workspace)
        assert 'files' not in result['outgoing_knowledge']
        assert result['research_organization']['group'] == group
        for filename, value in [('active.json', result), ('training.json', result),
                                ('archive.json', result['outgoing_knowledge'])]:
            write_json(agent.workspace / filename, value)
            assert trace not in (agent.workspace / filename).read_text()
        assert (agent.workspace / TRACE).is_symlink()
        assert (agent.workspace / TRACE).resolve() == references[TRACE]
        snapshot = retain_evaluated_python_model(agent.workspace, 'model.py', [], 0.)
        snapshots.append(Path(snapshot['workspace']))
    # Measure allocated blocks, including both actual workspaces and their
    # saved results. A second raw inherited trace would exceed this bound.
    increment = allocated(tmp_path) - fixed_allocation
    assert increment < len(trace.encode()) // 2
    assert (tmp_path / 'reference_archive' / TRACE).read_text() == trace
    for snapshot in snapshots:
        assert (snapshot / TRACE).resolve() == references[TRACE]
        assert not (snapshot / 'parameter.json').is_symlink()
        score = evaluate_python_model(snapshot, 'model.py', [Observation({'u': 2.}, 11.)])
        assert score['returncode'] == 0 and score['mse'] == 0
    (snapshots[0] / 'parameter.json').write_text('{"bias": 100}')
    assert json.loads((snapshots[1] / 'parameter.json').read_text()) == {'bias': 5}
    assert json.loads((agents[0].workspace / 'parameter.json').read_text()) == {'bias': 5}
    assert evaluate_python_model(snapshots[1], 'model.py', [Observation({'u': 2.}, 11.)])['mse'] == 0
    assert all(agent.costs['observation_cost'] == agent.costs['model_calls'] == 0 for agent in agents)


def test_workspace_archive_load_recovers_python_role_contexts_and_failed_sources(tmp_path):
    failed = 'archive/failed/_evaluated_model'
    helper = 'archive/role_only/_evaluated_model/helper.py'
    catalogue = [{'id': 'failed-prior', 'directory': failed, 'model': f'{failed}/bad.py',
                  'status': 'failed', 'failure': 'Retained unsuccessful source'}]
    knowledge = {'files': {'model.py': 'def predict(row): return row["features"]["u"]\n',
        f'{failed}/bad.py': 'def predict(row): return -50.\n',
        f'{failed}/notes.md': 'This prior failed; retain its evidence.',
        helper: 'GAIN = 9\n', 'source_catalog.json': json.dumps(catalogue)},
        'source_catalog': catalogue, 'research_roles': {'helper': helper},
        'lessons': ['Old failure remains unresolved.']}
    root = tmp_path / 'study'
    agent = agent_at(root / 'case' / 'archive', knowledge)
    agent.model = 'model.py'  # A fixture submission; no scientific confirmation is asserted.
    agent._prepare_program_state()
    result = agent.result('budget_exhausted')
    save_source(root, agent, result)
    exported, sources = load_research_archive(root)
    context = Path(sources[0]['directory'])
    for name, original in knowledge['files'].items():
        assert exported['files'][str(context / name)] == original
    entry, = exported['source_catalog']
    assert entry['status'] == 'budget_exhausted'
    assert entry['lessons'] == knowledge['lessons']
    assert exported['research_roles']['predictor'] == sources[0]['model']
    assert sources[0]['fresh_confirmation_passed'] is False
    target = tmp_path / 'consumer'
    for name, content in exported['files'].items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    score = evaluate_python_model(target, sources[0]['model'], [Observation({'u': 7.}, 7.)])
    assert score['returncode'] == 0 and score['mse'] == 0


def test_workspace_native_refs_expand_for_archive_and_restore_without_repurchase(tmp_path, monkeypatch):
    root = tmp_path / 'study'
    agent, lab = agent_at(root / 'case' / 'archive', {'files': {}}), PaidConditionalLab()
    buy_fixture(agent, lab)
    feedback = execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'research_question': {'id': 'typed-native', 'question': 'Compare fixture programs',
            'parent_id': 'task', 'depends_on': [], 'context_ref': CONTEXT, 'theory_target': TARGET},
        'max_compute': 500_000, 'max_candidates': 2,
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}]})
    assert 'error' not in feedback
    original = deepcopy(next(q for q in agent.questions if q['id'] == 'typed-native')[
        'native_condition_research'])
    result = agent.result('researching')
    question = next(q for q in result['online_questions'] if q['id'] == 'typed-native')
    assert question['native_condition_research']['full_content_file'] == 'research_state.json'
    direct = resolve_workspace_archive(result['outgoing_knowledge'])
    assert direct['theory_index'] == result['outgoing_knowledge']['theory_index']
    assert direct['question_traces'] == result['outgoing_knowledge']['question_traces']
    assert direct['workspace'] == str(agent.workspace)
    save_source(root, agent, result)
    exported, _ = load_research_archive(root)
    trace = next(q for q in exported['source_catalog'][0]['question_traces'] if q['id'] == 'typed-native')
    assert trace['native_condition_research'] == json.loads(json.dumps(original))
    restored, restored_lab = agent_at(tmp_path / 'restored', {'files': {}}), PaidConditionalLab()

    def no_acquisition(*args, **kwargs):
        raise AssertionError('A retained native bank must restore without repurchasing labels')

    monkeypatch.setattr(restored_lab, 'execute', no_acquisition)
    restored.restore_research(result, agent.workspace, restored_lab, confirmation_seed=19)
    restored_question = next(q for q in restored.questions if q['id'] == 'typed-native')
    assert restored_question['native_condition_research'] == json.loads(json.dumps(original))
    assert restored.costs == agent.costs
    assert restored.observations == agent.observations and restored_lab.paid == lab.paid
    assert restored.costs['model_calls'] == 0
