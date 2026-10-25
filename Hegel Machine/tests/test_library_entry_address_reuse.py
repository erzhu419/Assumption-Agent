"""Small source addresses preserve real paid-failure and PR2 basis behavior.

These regressions use the existing paid-failure and real PR2 basis fixtures;
they acquire no observations and execute no prediction, fitting, or model.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

from hegel_machine.cognition import question_routing
from hegel_machine.cognition import theory_consolidation as library
from test_frozen_counterexample_reuse import scene as failed_scene
from test_question_routing import future_question
from test_theory_consolidation import agent_scene


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def installed_failure(tmp_path):
    retained = failed_scene()
    agent, _ = agent_scene(tmp_path / 'target')
    agent.theory_index = deepcopy(retained.theory_index)
    source = tmp_path / 'source/shared/theory_library.json'
    view = tmp_path / 'frozen.json'
    raw = deepcopy(retained.theory_consolidation)
    # One actual paid failure may carry several witnesses. Their order and
    # original addresses must remain visible through the shared batch read.
    raw['transfers'][0]['checks'].append(deepcopy(raw['transfers'][0]['checks'][0]))
    raw['transfers'][0]['checks'][1]['source_ref']['field'] = 'counterexamples[1]'
    save(source, raw)
    save(view, {'source_ref': {'file': str(source), 'json_path': []}})
    reference = {'file': str(view), 'json_path': []}
    agent.theory_consolidation = library.install_theory_library(agent, reference)
    question = agent._open_question(future_question(agent, 'retained_failure_basis'))
    question['frozen_library_path'] = str(view)
    return agent, question, source, view, raw, reference


def test_repeated_real_basis_and_failure_batch_do_not_reopen_large_sources(tmp_path, monkeypatch):
    agent, question, source, view, raw, _ = installed_failure(tmp_path)
    expected = question_routing._basis(agent, question, {})
    failed = next(item for item in question_routing.resolved_theories(agent) if item['id'] == 'failed')
    original_refs = deepcopy(failed['scope_refutation']['counterexample_refs'])
    assert any(item.get('file') == str(source) for item in original_refs)
    binding = deepcopy(agent.theory_consolidation['transfers'][0]['binding'])
    documents = {}
    batch_refs = library._retained_binding_failures(agent, agent._theory_library_base,
        str(view), binding, documents=documents)
    assert [item['json_path'][-1] for item in batch_refs[:2]] == [0, 1]

    def forbidden(*args, **kwargs):
        raise AssertionError('unchanged repeated basis or failure batch reopened its source')

    monkeypatch.setattr(library, 'load_source_values', forbidden)
    for _ in range(3):
        assert question_routing._basis(agent, question, {}) == expected
        actual = next(item for item in question_routing.resolved_theories(agent) if item['id'] == 'failed')
        assert actual['scope_refutation']['counterexample_refs'] == original_refs
        assert library._retained_binding_failures(agent, agent._theory_library_base,
            str(view), binding, documents=documents) == batch_refs


def test_local_failure_source_changes_and_reinstall_keep_physical_addresses(tmp_path, monkeypatch):
    agent, question, source, view, raw, reference = installed_failure(tmp_path)
    old_reference = library.library_entry_reference(agent, 'transfers', 0, 'checks', 0)
    assert old_reference['file'] == str(source)
    reads = []
    original = library.load_source_values

    def read(references, workspace):
        reads.extend(item['file'] for item in references)
        return original(references, workspace)

    monkeypatch.setattr(library, 'load_source_values', read)
    unique = deepcopy(raw['transfers'][0])
    unique['theory_id'] = 'new_failed'
    unique['theory_ref']['id'] = 'new_failed'
    unique['checks'][0]['source_ref'] = {'file': '/retained/new_paid.json', 'json_path': ['rows', 7]}
    agent.theory_consolidation['transfers'].append(unique)
    assert library.library_entry_reference(agent, 'transfers', 1, 'checks', 0) == {
        'file': library.LIBRARY_PATH, 'json_path': ['transfers', 0, 'checks', 0]}
    assert reads == []
    assert agent.theory_consolidation['transfers'][1]['checks'][0]['source_ref'] == unique['checks'][0]['source_ref']

    # The old base value moves within the source. Its cached physical position
    # must change even though the live base's own original index stays zero.
    moved = deepcopy(raw)
    moved['transfers'].insert(0, unique)
    save(source, moved)
    moved_reference = library.library_entry_reference(agent, 'transfers', 0, 'checks', 0)
    assert moved_reference['file'] == str(source)
    assert moved_reference['json_path'] == ['transfers', 1, 'checks', 0]
    assert str(source) in reads
    reads.clear()
    assert library.library_entry_reference(agent, 'transfers', 0, 'checks', 0) == moved_reference
    assert reads == []

    # Changing a thin layer must follow its new actual source, rather than the
    # old dependency's otherwise unchanged metadata.
    replacement = tmp_path / 'other/shared/theory_library.json'
    save(replacement, raw)
    save(view, {'source_ref': {'file': str(replacement), 'json_path': []}})
    switched = library.library_entry_reference(agent, 'transfers', 0, 'checks', 0)
    assert switched['file'] == str(replacement)
    assert switched['json_path'] == ['transfers', 0, 'checks', 0]
    assert str(replacement) in reads

    agent.theory_consolidation = library.install_theory_library(agent, reference)
    reads.clear()
    # Reinstallation deliberately invalidates prior address lookups.
    assert library.library_entry_reference(agent, 'transfers', 0, 'checks', 0) == switched
    assert str(replacement) in reads
    reads.clear()
    assert library.library_entry_reference(agent, 'transfers', 0, 'checks', 0) == switched
    assert reads == []


def test_direct_revision_proposals_share_one_failure_source_read(tmp_path, monkeypatch):
    from hegel_machine.cognition import revision_relations
    from test_question_routing import CONTEXT, theory
    from test_revision_context_fitting import FITTING, examples
    from test_revision_relations import binary, constant, edit_candidate, feature

    agent, lab = agent_scene(tmp_path / 'target')
    candidate = {**edit_candidate(), 'version': 1}
    parent = binary('mul', constant(5.), feature())
    question_routing.register_theories(agent, [theory('parent', parent)])
    checks = [{**item, 'source_ref': {'file': str(tmp_path / 'paid_fixture.json'), 'json_path': [index]},
               'context_ref': deepcopy(CONTEXT)} for index, item in enumerate(examples(parent))]
    fitting = {**FITTING, 'operations': ['refit'], 'max_structures': 2}
    # Freeze actual proposals from the existing direct revision fixture so the
    # regression isolates their consumption, without inventing a repair Program.
    proposed = revision_relations.propose_revision(candidate, agent.theory_index['parent'], [],
        allocation={'compute': 5000}, evidence_checks=checks, fitting=fitting)
    assert len(proposed['proposals']) >= 2
    source = tmp_path / 'frozen/shared/theory_library.json'
    transfers = []
    for index, proposal in enumerate(proposed['proposals']):
        binding = {'library_path': str(source), 'candidate_id': candidate['id'],
            'candidate_version': 1, 'components': deepcopy(proposal['bindings']),
            'context_ref': deepcopy(proposal['context_ref'])}
        if proposal.get('parent_binding'):
            binding.update(parent=deepcopy(proposal['parent_binding']),
                           constraint_hypothesis=proposal['constraint_hypothesis'])
        if 'fitting' in proposal:
            binding['fitting'] = deepcopy(proposal['fitting'])
        transfers.append({'theory_id': f'retained_{index}',
            'theory_ref': {'id': f'retained_{index}', 'version': 1}, 'binding': binding,
            'status': 'refuted', 'checks': [{'status': 'failed',
                'context_ref': deepcopy(proposal['context_ref']),
                'source_ref': {'file': str(tmp_path / 'paid_failures.json'), 'json_path': [index]}}]})
    save(source, {'candidates': {candidate['id']: candidate}, 'transfers': transfers})
    monkeypatch.setattr(revision_relations, 'propose_revision', lambda *args, **kwargs: deepcopy(proposed))
    reads, relation_phase_reads, failure_phase_reads, returned = [], [], [], []
    original_read = library.load_source_values
    original_failures = library._retained_binding_failures
    original_relation = library._transfer_relation

    def read(references, workspace):
        reads.extend(item['file'] for item in references)
        return original_read(references, workspace)

    def failures(*args, **kwargs):
        before = len(reads)
        value = original_failures(*args, **kwargs)
        failure_phase_reads.extend(reads[before:])
        returned.append(deepcopy(value))
        return value

    def relation(*args, **kwargs):
        before = len(reads)
        value = original_relation(*args, **kwargs)
        relation_phase_reads.extend(reads[before:])
        return value

    monkeypatch.setattr(library, 'load_source_values', read)
    monkeypatch.setattr(library, '_retained_binding_failures', failures)
    monkeypatch.setattr(library, '_transfer_relation', relation)
    result = agent._execute('run', {'operator': 'theory_transfer',
        'library_path': str(source), 'candidate_id': candidate['id'], 'theory_id': 'direct_batch',
        'parent_theory_id': 'parent', 'context_ref': deepcopy(CONTEXT), 'output_unit': 'V',
        'allocation': {'compute': 5000}, 'evidence_checks': checks, 'fitting': fitting}, lab)
    assert len(result['generated_theories']) == len(proposed['proposals'])
    assert len(returned) == len(proposed['proposals'])
    # This counts the relation's constructor and proposal batch, excluding the
    # earlier ordinary source-library read in transfer_theory itself.
    assert relation_phase_reads.count(str(source)) == 1
    assert failure_phase_reads.count(str(source)) == 0
    assert returned == [[{'file': str(source), 'json_path': ['transfers', index, 'checks', 0]}]
                        for index in range(len(proposed['proposals']))]


def test_thin_layer_candidate_changes_keep_original_transfer_ordinals(tmp_path, monkeypatch):
    source = tmp_path / 'source/shared/theory_library.json'
    view = tmp_path / 'view.json'
    raw = deepcopy(failed_scene().theory_consolidation)
    raw['candidates'] = {'old': {'id': 'old', 'version': 1},
                         'removed': {'id': 'removed', 'version': 1}}
    kept = deepcopy(raw['transfers'][0])
    kept.update(theory_id='kept', theory_ref={'id': 'kept', 'version': 1})
    kept['binding']['parameters'] = [.7]
    raw['transfers'].append(kept)
    local = deepcopy(kept)
    local.update(theory_id='local', theory_ref={'id': 'local', 'version': 1})
    local['binding']['parameters'] = [.9]
    save(source, raw)
    save(view, {'source_ref': {'file': str(source), 'json_path': []},
        'removed_candidates': ['removed'], 'candidates': {'new': {'id': 'new', 'version': 1}},
        'removed_transfer_indices': [0], 'transfers': [local]})
    agent = SimpleNamespace(workspace=tmp_path / 'target')
    documents = {}
    selected = library.read_theory_library(agent, str(view), source_documents=documents)
    source_location = next(location for location in documents if location[0] == str(source))
    assert list(selected['candidates']) == ['old', 'new']
    assert [item['theory_id'] for item in selected['transfers']] == ['kept', 'local']
    assert [item['theory_id'] for item in documents[source_location]['transfers']] == ['failed', 'kept']
    # The small ordinal list holds the already loaded record itself.
    assert documents[source_location]['transfers'][1] is selected['transfers'][0]
    proxy = SimpleNamespace(workspace=agent.workspace, theory_consolidation=selected,
        _theory_library_base=selected, _theory_library_source_ref=library._library_reference(agent, str(view)))

    def forbidden(*args, **kwargs):
        raise AssertionError('original constructor/transfer provenance was already loaded')

    monkeypatch.setattr(library, 'load_source_values', forbidden)
    assert library.library_entry_reference(proxy, 'candidates', 'old', documents=documents) == {
        'file': str(source), 'json_path': ['candidates', 'old']}
    assert library.library_entry_reference(proxy, 'candidates', 'new', documents=documents) == {
        'file': str(view), 'json_path': ['candidates', 'new']}
    assert library.library_entry_reference(proxy, 'transfers', 0, documents=documents) == {
        'file': str(source), 'json_path': ['transfers', 1]}
    assert library.library_entry_reference(proxy, 'transfers', 1, documents=documents) == {
        'file': str(view), 'json_path': ['transfers', 0]}
    assert library._retained_binding_failures(agent, selected, str(view), kept['binding'],
        documents=documents) == [{'file': str(source), 'json_path': ['transfers', 1, 'checks', 0]}]


def test_new_failure_addresses_are_field_local_fresh_and_keep_first_duplicate_ordinal(tmp_path, monkeypatch):
    from test_frozen_counterexample_reuse import declaration

    agent, _, source, _, _, _ = installed_failure(tmp_path)
    question_routing.resolved_theories(agent)  # Retain the unchanged original failure's address.
    original = agent.theory_consolidation['transfers'][0]
    additions = []
    for identifier in ('new_a', 'new_b'):
        failure = deepcopy(original)
        failure.update(theory_id=identifier, theory_ref={'id': identifier, 'version': 1})
        failure['checks'][0]['source_ref'] = {
            'file': '/retained/new_paid.json', 'json_path': ['rows', identifier]}
        additions.append(failure)
        question_routing.register_theories(agent, [declaration(identifier, compatible=False,
            support={'status': 'refuted', 'evidence': []},
            consolidation_ref=deepcopy(failure['binding']))])
    agent.theory_consolidation['transfers'].extend([*additions, deepcopy(additions[0])])

    def forbidden(*args, **kwargs):
        raise AssertionError('a local failure address must not compute or reread the whole library')

    monkeypatch.setattr(library, '_library_document', forbidden)
    monkeypatch.setattr(library, 'load_source_values', forbidden)
    expected = {'file': library.LIBRARY_PATH, 'json_path': ['transfers', 0, 'checks', 0]}
    assert library.library_entry_reference(agent, 'transfers', 1, 'checks', 0) == expected
    assert library.library_entry_reference(agent, 'transfers', 3, 'checks', 0) == expected
    assert library.library_entry_reference(agent, 'transfers', 2, 'checks', 0) == {
        'file': library.LIBRARY_PATH, 'json_path': ['transfers', 1, 'checks', 0]}
    resolved = next(item for item in question_routing.resolved_theories(agent) if item['id'] == 'new_a')
    local_refs = [item['json_path'] for item in resolved['scope_refutation']['counterexample_refs']
                  if item.get('file') == library.LIBRARY_PATH]
    assert local_refs == [['transfers', 0, 'checks', 0], ['transfers', 0, 'checks', 1],
                          ['transfers', 1, 'checks', 0], ['transfers', 1, 'checks', 1]]
    assert any(item.get('file') == str(source) for item in resolved['scope_refutation']['counterexample_refs'])


def test_real_scope_failure_batch_filters_local_field_once_and_next_batch_sees_edits(tmp_path):
    agent, _, _, _, _, _ = installed_failure(tmp_path)
    original = agent.theory_consolidation['transfers'][0]

    def failure(marker):
        record = deepcopy(original)
        record['checks'][0]['source_ref']['field'] = marker
        return record

    first, second = failure('local_first'), failure('local_second')

    class Transfers(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    transfers = Transfers([original, first, second, deepcopy(first)])
    agent.theory_consolidation['transfers'] = transfers

    def batch():
        transfers.iterations = 0
        actual = next(item for item in question_routing.resolved_theories(agent) if item['id'] == 'failed')
        # One ordered record traversal and one local-field delta per batch,
        # even with several failed records and multiple witnesses per record.
        assert transfers.iterations == 2
        return actual['scope_refutation']['counterexample_refs']

    refs = batch()
    local = [item['json_path'] for item in refs if item.get('file') == library.LIBRARY_PATH]
    assert local == [['transfers', 0, 'checks', 0], ['transfers', 0, 'checks', 1],
                     ['transfers', 1, 'checks', 0], ['transfers', 1, 'checks', 1]]
    # Replace the first local value without changing length. The equal old
    # value now lives at physical delta index 2, not its previous first index 0.
    transfers[1] = failure('local_replacement')
    fresh = batch()
    assert any(item.get('field') == 'local_replacement' for item in fresh)
    assert {'file': library.LIBRARY_PATH, 'json_path': ['transfers', 2, 'checks', 0]} in fresh
    transfers.append(failure('local_appended'))
    appended = batch()
    assert any(item.get('field') == 'local_appended' for item in appended)
    assert {'file': library.LIBRARY_PATH, 'json_path': ['transfers', 3, 'checks', 0]} in appended
