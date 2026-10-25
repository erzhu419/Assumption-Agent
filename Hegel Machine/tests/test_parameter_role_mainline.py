"""A real cross-source role mismatch changes the existing research obligation."""

from copy import deepcopy

from hegel_machine.cognition import revision_parameter_relations
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.revision_relations import consolidate_revision_relations
from test_revision_relations import binary, constant, feature, revision


def test_role_mismatch_stops_exact_enumeration_and_returns_existing_author_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unaligned exact targets must not reach structure enumeration')

    monkeypatch.setattr(revision_parameter_relations, 'iter_synthesis_structures', forbidden)
    sources = []
    for source_index in range(2):
        basis = feature() if source_index == 0 else Program('window_sum', lag=2, children=(feature(),))
        for index, value in enumerate((1.1, 2.2, 3.3)):
            before = binary('mul', constant(value), basis)
            after = binary('mul', constant(2. * value + .37), basis)
            sources.append(revision(f'{source_index}:{index}', after,
                source_id=f'physical_{source_index}', before_program=before.to_dict(),
                counterexamples=[{'source_ref': {'file': 'paid_role_fixture.json',
                    'json_path': [source_index, index]},
                    'observation': {'features': {'u': 1.}, 'target': 100.,
                        'history': [{'u': 1.}, {'u': 1.}]}, 'tolerance': 1e-10}]))
    original = deepcopy(sources)
    result = consolidate_revision_relations(sources, allocation={'compute': 100000},
                                           object_type='program_revision_relation')
    assert result['parameter_relation_searches'] and result['author_work']
    assert result['candidates'], 'The existing structural revisions remain available.'
    assert all(not candidate.get('parameter_relations') for candidate in result['candidates'])
    assert all(request['type'] == 'AuthorWorkRequest' for request in result['author_work'])
    assert any(request['reason'] == 'common_computational_parameter_role_required'
               for request in result['author_work'])
    assert not result['support_updated'] and result['new_observations'] == result['llm_calls'] == 0
    assert sources == original
