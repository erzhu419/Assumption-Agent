"""Restore the complete paid successor pool and its command/measurement pairs."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from scripts.retain_confirmed_structure import paid_support_mask, reference_scope_examples
from scripts.revise_learned_structure_successor import restore as restore_prior
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, plain, save
from scripts.run_partial_role_successors import knowledge_changes, apply_knowledge_changes
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import Observation, Probe
from hegel_machine.cognition.object_continuation import decode_entry, observations


def pair_rows(sequence, measured_rows):
    """Bind one complete probe, starting a fresh command history at its reset."""
    if len(sequence) != len(measured_rows):
        raise ValueError('Paid commands and measurements must cover the same trajectory')
    history, pairs = [], []
    for point, row in zip(sequence, measured_rows):
        pairs.append((Observation(dict(point), row.target, tuple(history)), row))
        history.append({**point, '__target__': row.target})
    return pairs


def restore_interval_scene(source):
    """Read the latest completed revision without repeating its scientific work."""
    source = Path(source)
    registration = load(source / 'registration.json')
    receipt = load(source / 'receipt.json')
    prior = restore_prior(registration['source'])
    selection = load(source / 'selection.json')
    if not selection['full_comparison']:
        raise ValueError('The retained interval pool has no complete paid comparison')
    pool, refs, old_scores = [], [], {}
    for record in selection['scores']:
        entry = decode_entry(read_method_reference(record['candidate_ref'], source))
        pool.append(entry)
        refs.append(record['candidate_ref'])
        old_scores[entry['candidate'].program] = record['loss']
    selected = decode_entry(read_method_reference(selection['selected_ref'], source))
    parent = next(item for item in pool if item['candidate'] == selected['candidate'])
    frozen = load(source / 'frozen_predictors.json')
    if (parent['candidate'].program != Program.from_dict(frozen['selected'])
            or frozen['fallback'] != prior['frame']['fallback']):
        raise ValueError('The saved parent and paid prediction contract differ')
    parent['fallback'] = frozen['fallback']
    latest = load(source / 'confirmation_observations.json')
    exposed = observations(latest['batch']['observations'])
    exposed_scores = {Program.from_dict(frozen[name]): receipt['confirmation'][name]['mse']
                      for name in ('selected', 'incumbent')}
    exposed_predictions = {Program.from_dict(frozen[name]): receipt['confirmation'][name]['predictions']
                           for name in ('selected', 'incumbent')}
    paid_before = prior['paid']
    paid = [*paid_before, *exposed]
    paired_refs = [*prior['scene']['references']['training'],
        *prior['scene']['references']['validation'],
        ref(Path(registration['source']) / 'confirmation/observations.json'),
        ref(source / 'confirmation_observations.json')]
    paired_batches = []
    for location in paired_refs:
        document = read_method_reference(location, source)
        batch = document.get('batch', document)
        probe = batch.get('probe') or document['probe']
        measured = observations(batch['observations'])
        paired_batches.append(pair_rows(probe['sequence'], measured))
    if [row for batch in paired_batches for _, row in batch] != paid:
        raise ValueError('Paid command pairs do not match the complete retained row order')
    previous_costs = {'compute_cost': receipt['cumulative_compute'],
        'observation_cost': registration['previous_costs']['observation_cost'] + receipt['observation_cost'],
        'measurement_cost': registration['previous_costs']['measurement_cost'] + receipt['measurement_cost']}
    return {'agent': prior['agent'], 'spec': prior['scene']['spec'], 'frame': prior['frame'],
        'pool': pool, 'refs': refs, 'parent': parent, 'paid': paid, 'paid_before': paid_before,
        'exposed': exposed, 'old_scores': old_scores, 'exposed_scores': exposed_scores,
        'exposed_predictions': exposed_predictions, 'paired_batches': paired_batches,
        'paired_batch_refs': paired_refs,
        'transitions': [*prior['transitions'], *latest['batch'].get('transitions', [])],
        'original_registration': prior['registration'], 'registration': registration,
        'previous_costs': previous_costs,
        'remaining_compute': registration['total_compute_limit'] - previous_costs['compute_cost'],
        'source_refs': {'selection': ref(source / 'selection.json'),
            'scores': ref(source / 'selection.json', 'scores'),
            'selected': selection['selected_ref'], 'receipt': ref(source / 'receipt.json'),
            'confirmation': ref(source / 'receipt.json', 'confirmation'),
            'paired_trajectories': paired_refs, 'old_source_costs': receipt['source_cost_refs']}}


def score_retained(scene, budget):
    """Reuse complete old scores and pay only for missing newly exposed rows."""
    old_rows, new_rows = len(scene['paid_before']), len(scene['exposed'])
    programs = list(dict.fromkeys(item['candidate'].program for item in scene['pool']))
    exposed = dict(scene['exposed_scores'])
    missing = [program for program in programs if program not in exposed]
    if not budget.can_spend(len(missing) * new_rows):
        raise RuntimeError('The complete retained interval comparison is not funded')
    before = budget.spent
    for program in missing:
        budget.spend(new_rows)
        exposed[program] = loss(program, scene['exposed'], scene['parent']['fallback'],
                                scene['spec'].risk_quantile)
    scores = {program: (old_rows * scene['old_scores'][program] + new_rows * exposed[program])
                       / (old_rows + new_rows) for program in programs}
    return scores, budget.spent - before


def restore_interval_result(output):
    """Resume a completed interval attempt, including useful unconfirmed research."""
    output = Path(output)
    registration = load(output / 'registration.json')
    receipt = load(output / 'receipt.json')
    selection = load(output / 'selection.json')
    if not selection['full_comparison']:
        raise ValueError('The interval result has no complete paid comparison')
    base = restore_interval_scene(registration['source'])
    pool, refs, scores = [], [], {}
    for record in selection['scores']:
        entry = decode_entry(read_method_reference(record['candidate_ref'], output))
        pool.append(entry)
        refs.append(record['candidate_ref'])
        scores[entry['candidate'].program] = record['loss']
    selected = decode_entry(read_method_reference(selection['selected_ref'], output))
    parent = next(item for item in pool if item['candidate'] == selected['candidate'])
    parent.update(selected)
    parent['fallback'] = selected.get('fallback', base['parent']['fallback'])
    paid, pairs = list(base['paid']), list(base['paired_batches'])
    experiments = [tuple(base['exposed'])]
    transitions = list(base['transitions'])
    new_refs = selection['new_observation_refs']
    for location in new_refs:
        document = read_method_reference(location, output)
        batch = document.get('batch', document)
        probe = batch.get('probe') or document['probe']
        rows = observations(batch['observations'])
        paid.extend(rows)
        experiments.append(tuple(rows))
        pairs.append(pair_rows(probe['sequence'], rows))
        transitions.extend(batch.get('transitions', []))
    agent = base['agent']
    agent.library.instrument_models = load(output / 'instrument_models.json')
    agent.library.program_experiment_methods.records = load(output / 'program_experiment_methods.json')
    pending_ref = None
    for action in selection['actions']:
        if action['action'] == 'plan_relation_experiment':
            pending_ref = action['plan_ref']
        elif action['action'] == 'test_relation_experiment':
            pending_ref = None
    pending_plan = read_method_reference(pending_ref, output) if pending_ref else None
    if pending_plan and pending_plan['probe']:
        probe = pending_plan['probe']
        pending_plan['probe'] = Probe(probe['probe_id'], tuple(probe['sequence']))
    costs = {'compute_cost': receipt['cumulative_compute'],
        'observation_cost': registration['previous_observations'] + receipt['observation_cost'],
        'measurement_cost': registration['previous_measurement_cost'] + receipt['measurement_cost']}
    return {'agent': agent, 'spec': base['spec'], 'frame': base['frame'],
        'original_registration': base['original_registration'], 'registration': registration,
        'pool': pool, 'refs': refs, 'parent': parent, 'scores': scores,
        'paid': paid, 'paid_errors': selection['selected_paid_errors'],
        'experiments': experiments, 'pending_plan': pending_plan,
        'pending_plan_ref': pending_ref, 'calibration': agent.library.instrument_models[-1],
        'knowledge_source': registration['source'], 'knowledge_observation_refs': list(new_refs),
        'paired_batches': pairs, 'paired_batch_refs': [*base['paired_batch_refs'], *new_refs],
        'transitions': transitions, 'previous_costs': costs,
        'remaining_compute': registration['total_compute_limit'] - costs['compute_cost'],
        'stop_reason': receipt['stop_reason'], 'confirmation': receipt.get('confirmation'),
        'intervals': receipt['radius_intervals'], 'source_scene': base,
        'source_refs': {'selection': ref(output / 'selection.json'),
            'scores': ref(output / 'selection.json', 'scores'), 'selected': selection['selected_ref'],
            'receipt': ref(output / 'receipt.json'), 'new_observations': new_refs,
            'instrument_models': ref(output / 'instrument_models.json'),
            'program_experiment_methods': ref(output / 'program_experiment_methods.json'),
            'old_source_costs': receipt['source_cost_refs']}}


def save_confirmed_knowledge(agent, selected, spec, paid_rows, paid_errors, confirmation,
                             output, *, source, new_observation_refs, confirmation_ref=None):
    """Retain the confirmed answer and the research knowledge used to obtain it."""
    if not (confirmation or {}).get('selected', {}).get('passed'):
        return {'stored': False, 'reason': 'confirmation_not_passed'}
    source, output = Path(source), Path(output)
    if confirmation_ref is None:
        confirmation_ref = ref(output / 'receipt.json', 'confirmation')
    base = restore_interval_scene(source)
    before = plain(base['agent'].library.to_dict())
    retained_rows, batches = list(base['paid']), []
    for location in [*base['paired_batch_refs'], *new_observation_refs]:
        document = read_method_reference(location, output)
        batch = document.get('batch', document)
        batches.append({'probe': batch.get('probe') or document['probe'],
                        'channels': batch['channels']})
    for location in new_observation_refs:
        document = read_method_reference(location, output)
        retained_rows.extend(observations(document.get('batch', document)['observations']))
    if retained_rows != list(paid_rows):
        raise ValueError('Retained support must match the paid source and research trajectories')
    mask = paid_support_mask(selected['candidate'].program, paid_rows, paid_errors)
    recorded = agent.remember_confirmed_selection(selected, agent._knowledge_spec(spec),
        paid_rows, [], confirmed=True, confirmation_loss=confirmation['selected']['mse'],
        confirmation_rows=confirmation['rows'], selected_channels=('fine',), batches=batches,
        confirmation_ref=confirmation_ref, paid_support_matches=mask)
    if not recorded['stored']:
        return recorded
    after = plain(agent.library.to_dict())
    changes = deepcopy(knowledge_changes(before, after))
    support_refs = reference_scope_examples(changes, paid_rows, mask)
    save(output / 'knowledge_delta.json', {'source': str(source), 'changes': changes,
        'support_example_refs': support_refs, 'new_observation_refs': list(new_observation_refs),
        'confirmation_ref': confirmation_ref})
    restored = load_confirmed_knowledge(output)
    if plain(restored.to_dict()) != after:
        raise ValueError('Confirmed knowledge did not recover its complete live state')
    return {**recorded, 'knowledge_ref': ref(output / 'knowledge_delta.json'),
        'support_rows': sum(mask), 'confirmation_rows': confirmation['rows'],
        'support_examples_saved_by_reference': len(support_refs), 'readback_equal': True,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0}


def load_confirmed_knowledge(output):
    """Read the knowledge delta, resolving support rows from existing purchases."""
    output = Path(output)
    saved = load(output / 'knowledge_delta.json')
    base = restore_interval_scene(saved['source'])
    knowledge = apply_knowledge_changes(plain(base['agent'].library.to_dict()), saved['changes'])
    rows = list(base['paid'])
    for location in saved['new_observation_refs']:
        document = read_method_reference(location, output)
        rows.extend(observations(document.get('batch', document)['observations']))
    for binding in saved['support_example_refs']:
        knowledge['concepts'][binding['concept_index']]['scopes'][binding['scope_index']][
            'support_examples'] = plain([asdict(rows[index]) for index in binding['paid_row_indices']])
    return KnowledgeLibrary.from_dict(knowledge)


def load_repaired_knowledge(knowledge_root, repaired_entry_ref):
    """Rebind a paid method repair while retaining the confirmed world answer."""
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    root = Path(knowledge_root)
    saved = load(root / 'knowledge_delta.json')
    receipt = load(root / 'receipt.json')
    selected_ref = load(root / 'selection.json')['selected_ref']
    selected = decode_entry(read_method_reference(selected_ref, root))
    repaired = decode_entry(read_method_reference(repaired_entry_ref, root))
    original_program = selected['candidate'].program.expanded().simplified()
    if repaired['candidate'].program.expanded().simplified() != original_program:
        raise ValueError('Method repair changes the frozen selected world answer')
    library = load_confirmed_knowledge(root)
    concept_id = receipt['knowledge']['concept_id']
    concept = next(item for item in library.concepts if item.concept_id == concept_id)
    if concept.program.expanded().simplified() != original_program:
        raise ValueError('Frozen knowledge and selected world answer differ')
    prior_index = next(index for index, item in enumerate(library.meta_priors)
                       if item['source_concept_id'] == concept_id)
    prior = library.meta_priors[prior_index]
    # These fields concern the unchanged world answer. A carrier repair does
    # not acquire another confirmation or support a new transfer condition.
    for key in ('status', 'confirmation_loss', 'confirmation_rows', 'confirmation_ref'):
        if key in prior.get('research_program', {}):
            repaired['research_program'][key] = deepcopy(prior['research_program'][key])
    source = restore_interval_scene(saved['source'])
    agent = UnifiedResearchAgent(source['agent'].config, library=library)
    actions = []
    if not agent._remember_relation_research(prior, repaired, source['spec'], actions):
        raise ValueError('Repaired relation method has no complete measured-input binding')
    return library, {'source_concept_id': concept_id, 'prior_index': prior_index,
        'knowledge_ref': ref(root / 'knowledge_delta.json'),
        'selected_ref': selected_ref, 'repaired_entry_ref': deepcopy(repaired_entry_ref),
        'actions': actions, 'world_program_unchanged': True,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0,
        'source_cost_ref': ref(root / 'receipt.json')}
