"""Native method calls fund parameter research and expose only development feedback."""

from types import SimpleNamespace

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from tests.test_native_decision_checkpoint import prepared_scene
from test_transfer_partitions import observations, proposal


def test_partition_quote_rebinds_current_features_and_preserves_comparison_reserve(monkeypatch):
    from hegel_machine.cognition import transfer_partitions

    agent, frame, _ = prepared_scene()
    payload = proposal()['candidates'][0]['research_program']
    payload['transfer_partition']['second_stage'] = {'compute_limit': 15000}
    template = SimpleNamespace(roles=('current_drive',), source_features=('drive',))
    plans = [('old_source', payload, template, [{'current_drive': Program('feature', name='new_drive')}])]
    calls = []

    def quote(bound, training, *, feedback_rows):
        calls.append((bound, training, feedback_rows))
        assert 'new_drive' in str(bound)
        return {'status': 'quoted', 'required_compute': 24000}

    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition', quote)
    assert agent._replay_search_allowance(plans, frame['training'], frame['comparison'], 30000) == 24000
    assert calls[0][1:] == (frame['training'], frame['comparison'])
    assert agent._replay_search_allowance(plans, frame['training'], frame['comparison'], 10000) == 9988
    assert agent._replay_search_allowance(frame['replay_plans'], frame['training'],
        frame['comparison'], 30000) == 4000
    assert 'new_drive' not in str(payload)


def test_native_partition_replay_receives_paid_feedback_without_replacing_the_old_pool(monkeypatch):
    from hegel_machine.cognition import research_programs, transfer_partitions

    agent, frame, _ = prepared_scene()
    old_pool = list(frame['assessed'])
    source, payload, template, mappings = frame['replay_plans'][0]
    payload['transfer_partition'] = {'kind': 'Pi1', 'second_stage': {'compute_limit': 15000}}
    called = []

    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition',
        lambda payload, training, **kw: {'status': 'quoted', 'required_compute': 20000})

    def replay(payload, training, budget, **kwargs):
        called.append(kwargs)
        assert budget.remaining == 20000
        assert budget.spend(12000)
        return {'candidate': Candidate('revised', Program('const', value=1.), 'test'),
                'kind': 'replay', 'training_loss': 1.}

    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    result = agent.execute_research_decision(frame, 'program_replay', SearchBudget(25000), [])
    assert called[0]['feedback_rows'] is frame['comparison']
    assert called[0]['tolerance'] == agent.config.minimum_improvement
    assert result['quote']['search_compute_limit'] == 20000
    assert result['quote']['maximum_new_candidates'] == 4
    assert result['claims'][:len(old_pool)] == old_pool
    assert result['new_claims'][0]['selection_basis'] == 'development_feedback'
    assert result['full_comparison'] and not result['independent_confirmation']


def test_ordinary_methods_do_not_receive_new_development_inputs(monkeypatch):
    from hegel_machine.cognition import research_programs

    agent, frame, _ = prepared_scene()
    called = []

    def replay(payload, training, budget, **kwargs):
        called.append(kwargs)
        return {'candidate': Candidate('old', Program('const', value=1.), 'test'),
                'kind': 'replay', 'training_loss': 1.}

    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    agent.execute_research_decision(frame, 'program_replay', SearchBudget(1000), [])
    assert 'feedback_rows' not in called[0] and 'tolerance' not in called[0]


def test_ordinary_research_does_not_credit_method_feedback_as_independent(monkeypatch):
    from hegel_machine.cognition import research_programs, transfer_partitions

    agent, frame, _ = prepared_scene()
    frame['directions'] = ['program_replay']
    for claim in frame['assessed']:
        claim['family_id'] = 'T03'
    frame['replay_plans'][0][1]['transfer_partition'] = {
        'kind': 'Pi1', 'second_stage': {'compute_limit': 15000}}
    monkeypatch.setattr(agent, 'prepare_research_decision', lambda *args, **kw: frame)
    monkeypatch.setattr(transfer_partitions, 'quote_transfer_partition',
        lambda payload, training, **kw: {'status': 'quoted', 'required_compute': 20000})

    def replay(payload, training, budget, **kwargs):
        assert budget.remaining == 20000
        assert kwargs['feedback_rows'] == frame['comparison']
        budget.spend(9000)
        return {'candidate': Candidate('development', Program('const', value=1.), 'test'),
                'kind': 'replay', 'family_id': 'T03', 'training_loss': 1.}

    monkeypatch.setattr(research_programs, 'replay_research_program', replay)
    actions = []
    result = agent._research_view(frame['spec'], frame['training'], frame['validation'],
        SearchBudget(25000), actions, [], ResearchGraph(), include_risk_revisions=False)
    revised = next(item for item in result if item['candidate'].candidate_id == 'development')
    assert revised['selection_basis'] == 'development_feedback'
    assert any(item.get('evidence_basis') == 'current_method_development_feedback' for item in actions)
    assert not any(item['action'] == 'research_action_feedback' for item in actions)


def test_real_parameter_method_is_not_cut_to_four_thousand_compute():
    agent, frame, _ = prepared_scene()
    payload = proposal()['candidates'][0]['research_program']
    rows = observations(count=96)
    template = SimpleNamespace(roles=(), source_features=())
    plans = [('old_source', payload, template, [{}])]
    allowance = agent._replay_search_allowance(plans, rows, rows[:12], 40000)
    assert allowance > 4000
    work = SearchBudget(allowance)
    claims = agent._replay_program_claims(frame['spec'], rows, work, [], prepared=plans)
    assert claims and claims[0]['training_loss'] < 1e-12
    assert claims[0]['transfer_partition_search']['completed']
    assert work.spent <= allowance
