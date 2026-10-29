#!/usr/bin/env python3
"""Save a paid confirmed structural method; never repeat its source research."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

from scripts.partial_role_paid_scene import load_paid_scene
from scripts.run_partial_role_parent import load, save, plain
from scripts.run_partial_role_successors import (
    historical_knowledge, knowledge_changes, apply_knowledge_changes,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry, restore_object_state
from hegel_machine.cognition.structural_research_program import attach_structural_research_program
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ROOT = Path(__file__).resolve().parents[1]


def ref(path, *keys):
    return {'file': str(path), **({'json_path': list(keys)} if keys else {})}


def source_inputs(source):
    config = load(source / 'registration.json')
    receipt = load(source / 'receipt.json')
    selected = decode_entry(load(source / 'selected_entry.json'))
    frozen = load(source / 'frozen_predictor.json')
    selection = load(source / 'selection.json')
    if (receipt['status'] != 'confirmed' or not receipt['confirmation']['passed']
            or not selection['fresh_confirmation_admitted']
            or selected['candidate'].program.to_dict() != frozen['program']):
        raise ValueError('The selected structural answer has no matching paid confirmation')
    scene = load_paid_scene(config['root'], config['acceptance'], config['previous'])
    return config, receipt, selected, selection, scene


def paid_support_mask(program, rows, errors):
    """Use a sufficient part of the existing prediction_matches tolerance."""
    if len(rows) != len(errors):
        raise ValueError('Paid residuals must cover the same complete source rows')
    # Exact signs of the saved squared errors are unnecessary here: every
    # retained error is below the tolerance even without the prediction term.
    if any(sqrt(error) > 1e-6 * max(1., abs(row.target)) for row, error in zip(rows, errors)):
        raise ValueError('Saved squared residuals do not resolve all support matches')
    return [len(row.history) >= program.minimum_history_depth for row in rows]


def reference_scope_examples(changes, rows, mask):
    """The new scope borrows its paid trajectories instead of exporting them again."""
    examples = [asdict(row) for row, supported in zip(rows, mask) if supported]
    examples = plain(examples)
    references = []
    for index, concept in changes.get('concepts', {}).get('updated', {}).items():
        for position, scope in enumerate(concept.get('scopes', [])):
            if scope.get('support_examples') == examples:
                scope.pop('support_examples')
                references.append({'concept_index': int(index), 'scope_index': position,
                    'paid_row_indices': [i for i, supported in enumerate(mask) if supported]})
    return references


def load_retained_knowledge(output):
    saved = load(output / 'knowledge_delta.json')
    old, _, _ = historical_knowledge(saved['historical_knowledge_ref'])
    knowledge = apply_knowledge_changes(old, saved['changes'])
    if saved['support_example_refs']:
        config = load(Path(saved['source']) / 'registration.json')
        scene = load_paid_scene(config['root'], config['acceptance'], config['previous'])
        rows = scene['paid_rows']
        for binding in saved['support_example_refs']:
            knowledge['concepts'][binding['concept_index']]['scopes'][binding['scope_index']][
                'support_examples'] = plain([asdict(rows[i]) for i in binding['paid_row_indices']])
    return KnowledgeLibrary.from_dict(knowledge)


def retain(source, output):
    if (output / 'receipt.json').exists():
        return load(output / 'receipt.json')
    output.mkdir(parents=True, exist_ok=True)
    timing = ExecutionTiming(('restore', 'save_knowledge', 'readback'))
    with timing.phase('restore'):
        config, receipt, selected, selection, scene = source_inputs(source)
        knowledge, _, _ = historical_knowledge(config['historical_knowledge_ref'])
        agent = UnifiedResearchAgent(ResearchConfig(seed=config['seed'], max_history=4,
            minimum_improvement=config['tolerance'], compute_budget=config['compute_limit_including_previous']),
            library=KnowledgeLibrary.from_dict(knowledge))
        restore_object_state(agent, scene['report'], ('fine',))
        before = plain(agent.library.to_dict())
        rows = scene['paid_rows']
        selected = attach_structural_research_program(selected, fit_rows=len(rows),
            source_ref=ref(source / 'selected_entry.json'))
        mask = paid_support_mask(selected['candidate'].program, rows, selection['selected_paid_errors'])
        batches = []
        for location in scene['source_refs']['paid_trajectories']:
            document = load(Path(location['file']))
            batch = document
            for key in location['json_path']:
                batch = batch[key]
            probe = batch['probe'] if 'probe' in batch else document['probe']
            batches.append({'probe': probe, 'channels': batch['channels']})
        settings = next(item for item in scene['registration']['tasks'] if item['seed'] == config['seed'])
        with AnonymousObjectLab(**settings) as laboratory:
            spec = agent._knowledge_spec(agent._spec(laboratory, ('fine',)))
    confirmation = receipt['confirmation']
    with timing.phase('save_knowledge'):
        recorded = agent.remember_confirmed_selection(selected, spec, rows, [], confirmed=True,
            confirmation_loss=confirmation['selected_mse'], confirmation_rows=confirmation['rows'],
            selected_channels=('fine',), confirmation_ref=ref(source / 'receipt.json', 'confirmation'),
            batches=batches,
            paid_support_matches=mask)
        if not recorded['stored']:
            raise ValueError('The original confirmation did not admit the structural knowledge')
        after = plain(agent.library.to_dict())
        changes = deepcopy(knowledge_changes(before, after))
        support_refs = reference_scope_examples(changes, rows, mask)
        save(output / 'retained_entry.json', encode_entry(selected))
        save(output / 'knowledge_delta.json', {'source': str(source),
            'historical_knowledge_ref': config['historical_knowledge_ref'], 'changes': changes,
            'support_example_refs': support_refs, 'source_rows': scene['source_refs']['paid_trajectories']})
    with timing.phase('readback'):
        restored = load_retained_knowledge(output)
        if plain(restored.to_dict()) != after:
            raise ValueError('The reference-backed knowledge did not recover the exact live result')
        agent.library = restored
        calls = agent._recall(spec, rows)
        plans = agent._replay_program_queue(spec, calls, rows)
        accepted = [item for item in plans if item[0] == recorded['concept_id']]
        if not accepted:
            raise ValueError('The new confirmed program is absent from the ordinary replay queue')
        repeated = agent.remember_confirmed_selection(selected, spec, rows, [], confirmed=True,
            confirmation_loss=confirmation['selected_mse'], confirmation_rows=confirmation['rows'],
            confirmation_ref=ref(source / 'receipt.json', 'confirmation'), paid_support_matches=mask)
        if not repeated.get('reused') or plain(agent.library.to_dict()) != after:
            raise ValueError('A repeated paid confirmation changed retained knowledge')
    result = {'status': 'confirmed_structure_method_retained', 'source_ref': ref(source / 'receipt.json'),
        'selected_ref': ref(output / 'retained_entry.json'), 'knowledge_ref': ref(output / 'knowledge_delta.json'),
        'stored': recorded, 'old_concepts': len(before['concepts']), 'concepts': len(after['concepts']),
        'old_priors': len(before['meta_priors']), 'priors': len(after['meta_priors']),
        'fit_rows': len(rows), 'support_rows': sum(mask), 'confirmation': {
            'rows': confirmation['rows'], 'loss': confirmation['selected_mse'], 'already_paid': True},
        'term_count': len(selected['research_program']['term_nodes']),
        'structural_replay': selected['research_program']['structural_replay'],
        'ordinary_replay_sources': [item[0] for item in plans],
        'source_bindings': len(accepted[0][3]), 'readback_equal': True,
        'duplicate_confirmation_reused': True, 'support_examples_saved_by_reference': len(support_refs),
        'new_compute': 0, 'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'model_calls': 0,
        'replay_executed': False, 'independent_method_gain': None, 'execution_timing': timing.result(),
        'scope': 'paid source knowledge retention and ordinary recall; successor efficacy untested'}
    save(output / 'receipt.json', result)
    return result


def prepare(source, output):
    source_inputs(source)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {'source': str(source), 'scope': 'retain_paid_confirmed_method',
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0,
        'stop_rule': 'retain any failure; no automatic successor or renewed source research'})
    names = [p.relative_to(ROOT) for p in (ROOT / 'src/hegel_machine').rglob('*')
             if p.is_file() and p.suffix in ('.py', '.json')]
    names.extend(Path('scripts') / name for name in ('retain_confirmed_structure.py',
        'partial_role_paid_scene.py', 'run_partial_role_parent.py', 'run_partial_role_successors.py'))
    for name in names:
        target, base = output / 'runtime' / name, source / 'runtime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_incrementally(target, (ROOT / name).read_text(), base=base if base.exists() else None)


def run(output):
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Run this newly frozen current entry')
    source = Path(load(output / 'registration.json')['source'])
    if (output / 'failure.json').exists():
        raise ValueError('Retain the failed unit; diagnose it before another frozen attempt')
    def no_science(*args, **kwargs):
        raise AssertionError('Paid method retention cannot repeat fits, predictions or acquisition')
    try:
        with ExitStack() as stack:
            for name in ('hegel_machine.cognition.language.Program.predict',
                         'hegel_machine.cognition.research_programs._Fitter.fit',
                         'hegel_machine.cognition.anonymous_object_lab.AnonymousObjectLab.acquire'):
                stack.enter_context(patch(name, no_science))
            return retain(source, output)
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'source': str(source),
            'scope': 'knowledge retention; fitting, prediction and acquisition disabled'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source.resolve(), args.output.resolve()) if args.mode == 'prepare'
          else run(args.output.resolve()))
