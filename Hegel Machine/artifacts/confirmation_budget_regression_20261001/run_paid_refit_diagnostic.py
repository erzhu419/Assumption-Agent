"""One already-seen root03/stage8 replay from its exact paid incoming library."""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_knowledge_lineage_study import read, save, load_runtime
from run_research_investment_attribution_study import laboratory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / args.label
    if (output / 'summary.json').exists():
        raise SystemExit('This diagnostic label already has results; preserve it.')
    study = ROOT / 'artifacts/research_investment_attribution_20261001/fresh_replication'
    prefix_path = study / 'runs/candidate/root03/visit08.json.gz'
    prefix = read(prefix_path)
    task = read(study / 'protocol.json')['roots']['root03']['tasks'][8]
    module = load_runtime({'arms': {'diagnostic': {'runtime': str(args.runtime)}}}, 'diagnostic')
    from hegel_machine import __version__
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    library = KnowledgeLibrary.from_dict(prefix['knowledge'])
    limits = task['research_limits']
    config = replace(module.ResearchConfig(**prefix['config']), seed=task['seed'],
        observation_budget=limits['observation_budget'], compute_budget=limits['compute_budget'])
    cpu, wall = time.process_time(), time.monotonic()
    with laboratory(task) as lab:
        try:
            report = UnifiedResearchAgent(config, measurement_budget=limits['measurement_budget'],
                library=library).run(lab)
        except Exception as error:
            save(output / 'failure.json', {'error': repr(error), 'costs': lab.costs,
                'cpu_seconds': time.process_time() - cpu, 'incoming': str(prefix_path)})
            raise
    cpu, wall = time.process_time() - cpu, time.monotonic() - wall
    save(output / 'report.json.gz', {'version': __version__, 'task': task,
        'incoming': str(prefix_path), 'research': report, 'knowledge': library.to_dict(),
        'cpu_seconds': cpu, 'elapsed_seconds': wall, 'terminal_scoring': False})
    report = read(output / 'report.json.gz')['research']
    old = {arm: read(study / 'runs' / arm / 'root03/visit09.json.gz')['results'][8]['research']
        for arm in ('baseline_156', 'candidate')}
    action_counts = Counter(a['action'] for a in report['actions'])
    old_counts = Counter(a['action'] for a in old['candidate']['actions'])
    summary = {'version': __version__, 'label': args.label, 'scope': 'seen-case causal diagnosis, no new generalization',
        'root': 'root03', 'position': 8, 'task': task, 'incoming': str(prefix_path),
        'status': report['status'], 'selected_kind': report['selected_kind'], 'costs': report['costs'],
        'cpu_seconds': cpu, 'elapsed_seconds': wall, 'confirmation_loss': report['confirmation_loss'],
        'selection_loss': report['selection_loss'], 'used_sources': report['used_sources'],
        'concepts_added': report.get('concepts_added', []),
        'same_paid_batches_as_157_1': report['batches'] == old['candidate']['batches'],
        'program_equal_157_1': report['program'] == old['candidate']['program'],
        'program_equal_156': report['program'] == old['baseline_156']['program'],
        'actions': dict(action_counts), 'action_count_changes_vs_157_1': {
            key: action_counts[key] - old_counts[key] for key in action_counts.keys() | old_counts.keys()
            if action_counts[key] != old_counts[key]},
        'paid_revision_actions': [a for a in report['actions'] if 'paid' in a['action'] or
            a['action'] == 'defer_confirmation_revision'],
        'terminal_scoring': False, 'external_model_calls': 0}
    save(output / 'summary.json', summary)
    print({key: summary[key] for key in ('version', 'status', 'selected_kind', 'costs', 'cpu_seconds',
        'confirmation_loss', 'same_paid_batches_as_157_1', 'program_equal_157_1', 'program_equal_156')}, flush=True)


if __name__ == '__main__':
    main()
