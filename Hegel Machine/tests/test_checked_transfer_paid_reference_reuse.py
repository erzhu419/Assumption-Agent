"""Ordinary Common comparison facts survive a later empty-check transfer."""
from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition import online_inquiry_llm, research_programs
from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
import test_revision_comparison_failures as comparison


def test_common_refutation_blocks_old_empty_candidate_and_batches_paid_references(tmp_path, monkeypatch):
    checked = consolidation._checked_transfer
    empty_records = []

    # This is the actual first-use/checked-use sequence: the first fit can be
    # registered before any point checks; its later full comparison is a fact
    # about that same retained instance, not permission to reuse the old status.
    def initial_unchecked(agent, arguments, candidate, program, binding, **kwargs):
        initial = checked(agent, {**arguments, 'evidence_checks': []}, candidate,
                          program, binding, **kwargs)
        assert initial['status'] == 'candidate_registered' and initial['checks'] == []
        empty_records.append(deepcopy(agent.theory_consolidation['transfers'][-1]))
        return checked(agent, arguments, candidate, program, binding, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(comparison, '_checked_transfer', initial_unchecked)
        agent, lab, owner, identifier, models = comparison.comparison_scene(tmp_path)
    before_comparison = deepcopy(agent.costs)
    result = comparison.compare(agent, lab, owner, models)
    assert result['revision_comparison_failures']['new_refutation_count'] == 1
    assert result['new_scored_count'] == 3 and result['reused_score_count'] == 1
    assert agent.costs['compute_cost'] - before_comparison['compute_cost'] == 3 * len(agent.rows)
    failure = deepcopy(agent.theory_consolidation['transfers'][-1])
    assert 'checks' not in failure and 'check_basis' not in failure
    assert failure['ordinary_comparison_failure'] and failure['retained_counterexample_refs']
    assert agent.theory_consolidation['transfers'][0] == empty_records[0]
    original_transfers = deepcopy(agent.theory_consolidation['transfers'])
    costs, pool = deepcopy(agent.costs), set(agent.theory_index)
    indexed = agent.theory_index[identifier]

    def forbidden(*args, **kwargs):
        raise AssertionError('paid reference reuse must not fit, predict, acquire, or expand the old pool')

    monkeypatch.setattr(Program, 'predict', forbidden)
    monkeypatch.setattr(research_programs, 'replay_research_program', forbidden)
    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model', forbidden)
    monkeypatch.setattr(lab, 'execute', forbidden)
    monkeypatch.setattr(consolidation, '_read_library', forbidden)
    resolved = consolidation.resolve_program_sources
    reads, batches = [], []
    read_text = Path.read_text

    def source_fields(owner, references, *args, **kwargs):
        batches.append(deepcopy(references))

        def read_once(path, *read_args, **read_kwargs):
            reads.append(path)
            return read_text(path, *read_args, **read_kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_text', read_once)
            return resolved(owner, references, *args, **kwargs)

    monkeypatch.setattr(consolidation, 'resolve_program_sources', source_fields)
    retry = checked(agent, {'theory_id': identifier,
        'role_signature': indexed['role_signature'], 'output_unit': indexed['output_unit'],
        'context_ref': indexed['context_ref'], 'evidence_checks': []},
        {'id': 'fixture'}, Program.from_dict(indexed['program']), failure['binding'],
        compute_allowance=0, persist=False)
    assert retry['status'] == 'refuted' and retry['checks'] == []
    assert not retry.get('cache_hit')
    assert retry['new_compute_cost'] == 0 and agent.costs == costs
    assert agent.theory_consolidation['transfers'][:len(original_transfers)] == original_transfers
    assert set(agent.theory_index) == pool and not agent.theory_index[identifier]['compatible']
    assert batches == [[failure['check_basis_ref'], failure['checks_ref']]]
    assert reads == [agent.workspace / consolidation.LIBRARY_PATH]
