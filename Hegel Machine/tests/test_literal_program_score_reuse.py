"""Closed literal exports retain scores when unconsumed workspace material changes.

The scoring worker is a stub: these checks exercise cache inputs and debits,
not model prediction, fitting, or scientific evidence.
"""
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_python_paid_score_reuse import ChoiceLab, choose, no_execution, scoring_costs


MODEL = 'frozen/_evaluated_model/candidate_0000_v0001.py'


def scene(tmp_path, monkeypatch):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=200, confirmation_rows=2,
            compute_budget=1000, execution_seconds=30., max_model_calls=0))
    agent._open_question({'id': 'q', 'question': 'Mechanism-only fixed paid score fixture',
                          'context_ref': {'scene': 'literal_export_fixture'}})
    agent.rows = [Observation({'u': float(index)}, 0.) for index in range(97)]
    agent._save(MODEL, export_temporal_program(Program('const', value=1.)))
    agent._save('frozen/_evaluated_model/params.json', '{"unused_fixture_parameter":0}')
    agent._save('frozen/_evaluated_model/model_observations.json', '[]')
    calls = []

    def worker(*args, **kwargs):
        calls.append(args[1])
        return {'returncode': 0, 'mse': 0., 'loss': 0.,
                'model_dependencies': [], 'research_bindings': [], 'research_prediction_count': 0}

    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', worker)
    lab = ChoiceLab()
    first = choose(agent, lab, [MODEL])
    assert first['new_scored_count'] == 1 and agent.costs['compute_cost'] == 97
    return agent, lab, calls


def test_nineteen_sibling_exports_reuse_the_existing_97_row_score(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    for index in range(19):
        agent._save(f'frozen/_evaluated_model/candidate_{index:04d}_v0002.py',
                    export_temporal_program(Program('const', value=float(index))))
    before = scoring_costs(agent)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    repeated = choose(agent, lab, [MODEL])
    assert repeated['reused_score_count'] == 1 and repeated['new_scored_count'] == 0
    assert calls == [MODEL] and scoring_costs(agent) == before
    assert repeated['evaluations'][0]['paid_score_reuse']['additional_scoring_compute'] == 0


@pytest.mark.parametrize('change', ['selected_source', 'bindings', 'research_inputs'])
def test_literal_export_execution_inputs_still_invalidate_the_score(tmp_path, monkeypatch, change):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    directory = agent.workspace / 'frozen/_evaluated_model'
    if change == 'selected_source':
        (agent.workspace / MODEL).write_text(export_temporal_program(Program('const', value=2.)))
    elif change == 'bindings':
        (directory / 'model_bindings.json').write_text(json.dumps({
            'candidate_0000_v0001.py': {'child': 'another.py'}}))
    else:
        (directory / 'model_research_inputs.json').write_text(json.dumps({
            'candidate_0000_v0001.py': {'condition': {'version': 2}}}))
    repeated = choose(agent, lab, [MODEL])
    assert repeated['new_scored_count'] == 1 and repeated['reused_score_count'] == 0
    assert len(calls) == 2 and agent.costs['compute_cost'] == 194
    assert repeated['evaluations'][0]['paid_score_reuse']['reuse_miss_reason'] == 'python_execution_context_changed'


@pytest.mark.parametrize('renamed', [False, True])
def test_closed_literal_ignores_growing_research_json_and_unused_fit_view(tmp_path, monkeypatch, renamed):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    model = 'renamed/_evaluated_model/candidate.py' if renamed else MODEL
    agent._save(model, (agent.workspace / MODEL).read_text())
    directory = Path(model).parent
    for name, contents in (
        ('params.json', '{"unused_fixture_parameter":1}'),
        ('model_observations.json', '[{"features":{"u":1},"target":3}]'),
        ('native_representation_research/record_0000.json', '{"new_research":true}'),
        ('shared/theory_index.json', '{"new_candidates":[1,2]}'),
    ):
        agent._save(str(directory / name), contents)
    before = scoring_costs(agent)
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    original_read = Path.read_bytes

    def no_unused_json(path):
        if path.suffix == '.json':
            pytest.fail(f'closed predictor compared unconsumed JSON bytes: {path}')
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', no_unused_json)
    repeated = choose(agent, lab, [model])
    assert repeated['reused_score_count'] == 1 and repeated['new_scored_count'] == 0
    assert calls == [MODEL] and scoring_costs(agent) == before


def test_identical_renamed_literals_reuse_scores_and_index_once_per_comparison(tmp_path, monkeypatch):
    from hegel_machine.cognition import temporal_operator_agent as operators

    agent, lab, calls = scene(tmp_path, monkeypatch)
    unused = 'frozen/_evaluated_model/unused.py'
    agent._save(unused, export_temporal_program(Program('const', value=9.)))
    previous = choose(agent, lab, [unused])['evaluations'][0]['paid_score_reuse']
    assessment = json.loads(Path(previous['execution_basis_ref']['file']).read_text())
    unused_source = Path(assessment['execution_context']['workspace']) / 'unused.py'
    models = [f'frozen/_evaluated_model/renamed_{index}.py' for index in range(4)]
    for index, model in enumerate(models):
        agent._save(model, export_temporal_program(Program('const', value=1. if index < 2 else 2.)))
    read_text = Path.read_text
    reads = []

    def counted(path, *args, **kwargs):
        if path == unused_source:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', counted)
    before = agent.costs['compute_cost']
    compared = choose(agent, lab, models)
    # Two old-source aliases reuse the prior score; the first distinct source
    # scores once and its next alias reuses that result in this same comparison.
    assert compared['reused_score_count'] == 3 and compared['new_scored_count'] == 1
    assert agent.costs['compute_cost'] - before == 97
    assert calls == [MODEL, unused, models[2]]
    assert reads == [unused_source]  # No per-candidate rebuild of the old source index.
    assert [entry['path'] for entry in compared['evaluations']] == models
    assert all(entry['retention_decision']['status'] == 'incumbent_retained'
               for entry in compared['evaluations'])
    for index in (0, 1, 3):
        reuse = compared['evaluations'][index]['paid_score_reuse']
        assert reuse['match_basis'] == 'identical_closed_literal_execution'
        assert reuse['additional_scoring_compute'] == 0
        assert reuse['additional_scoring_python_seconds'] == 0.
    assert compared['evaluations'][0]['paid_score_reuse']['matched_model'] == MODEL
    assert compared['evaluations'][3]['paid_score_reuse']['matched_model'] == models[2]


@pytest.mark.parametrize('change', ['source', 'bindings', 'research_inputs', 'meaning', 'target', 'row_order'])
def test_renamed_literal_requires_same_source_context_and_scope(tmp_path, monkeypatch, change):
    from dataclasses import replace

    agent, lab, calls = scene(tmp_path, monkeypatch)
    model = 'renamed/_evaluated_model/candidate_v2.py'
    agent._save(model, (agent.workspace / MODEL).read_text())
    agent._save('renamed/_evaluated_model/params.json', '{"unused_fixture_parameter":0}')
    agent._save('renamed/_evaluated_model/model_observations.json', '[]')
    if change == 'source':
        agent._save(model, export_temporal_program(Program('const', value=2.)))
    elif change == 'bindings':
        agent._save('renamed/_evaluated_model/model_bindings.json',
                    '{"candidate_v2.py":{"child":"another.py"}}')
    elif change == 'research_inputs':
        agent._save('renamed/_evaluated_model/model_research_inputs.json',
                    '{"candidate_v2.py":{"condition":{"version":2}}}')
    elif change == 'meaning':
        active = next(question for question in agent.questions
                      if question['id'] == agent._active_question_id)
        active['meaning_version'] += 1
    elif change == 'target':
        agent.rows[0] = replace(agent.rows[0], target=2.)
    else:
        agent.rows.reverse()
    before = agent.costs['compute_cost']
    compared = choose(agent, lab, [model])
    assert compared['new_scored_count'] == 1 and compared['reused_score_count'] == 0
    assert agent.costs['compute_cost'] - before == 97 and calls == [MODEL, model]


def test_identical_general_python_at_different_paths_is_not_a_literal_score(tmp_path, monkeypatch):
    agent, lab, calls = scene(tmp_path, monkeypatch)
    source = 'def predict(row): return 1.\n'
    models = ['ordinary/_evaluated_model/first.py', 'ordinary/_evaluated_model/second.py']
    for model in models:
        agent._save(model, source)
    compared = choose(agent, lab, models)
    assert compared['new_scored_count'] == 2 and compared['reused_score_count'] == 0
    assert calls == [MODEL, *models]
