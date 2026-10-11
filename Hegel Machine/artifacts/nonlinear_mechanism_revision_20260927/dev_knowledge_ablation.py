#!/usr/bin/env python3
"""Development-only withdrawal of knowledge learned by the first deletion task."""

from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_nonlinear_mechanism_revision import laboratory
from run_knowledge_lineage_study import read, save, load_runtime


def main():
    output = Path(__file__).resolve().parent
    development = output / 'development'
    protocol = read(development / 'protocol.json')
    ablation_runtime = output / 'dev_knowledge_ablation' / 'runtime' / 'src'
    shutil.copytree(ROOT / protocol['arms']['mechanism_revision']['runtime'], ablation_runtime,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # All experiment code stays frozen. The sole change fixes without() dropping
    # unrelated stores. The source runtime is an independent ordinary copy.
    memory_path = Path('hegel_machine/cognition/memory.py')
    shutil.copy2(ROOT / 'src' / memory_path, ablation_runtime / memory_path)
    protocol['arms']['mechanism_revision']['runtime'] = str(ablation_runtime.relative_to(ROOT))
    runtime = load_runtime(protocol, 'mechanism_revision')
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import KnowledgeLibrary

    learned = []
    for root in protocol['roots']:
        directory = development / 'runs' / 'mechanism_revision' / root
        state = read(directory / 'visit02.json.gz')
        assert len(state['results']) == 2 and not state['pending_revisits']
        original = deepcopy(state['knowledge'])
        requested = set(state['results'][0]['research']['concepts_added'])
        assert requested and requested <= {item['concept_id'] for item in original['concepts']}
        wall, cpu = time.monotonic(), time.process_time()
        library = KnowledgeLibrary.from_dict(original)
        assert library.without(set()).to_dict() == library.to_dict()
        withdrawn = library.without(requested).to_dict()
        removed = set(withdrawn['deleted_ids']) - set(original['deleted_ids'])
        state['knowledge'] = KnowledgeLibrary.from_dict(withdrawn).to_dict()
        withdrawal = {
            'requested_concepts_from_first_report': sorted(requested),
            'dependency_closed_removed_ids': sorted(removed),
            'remaining_concepts': [item['concept_id'] for item in withdrawn['concepts']],
            'removed_meta_priors': [item['prior_id'] for item in original['meta_priors']
                if item not in withdrawn['meta_priors']],
            'retained_meta_priors': [item['prior_id'] for item in withdrawn['meta_priors']],
            'removed_definition_count': len(original['definitions']) - len(withdrawn['definitions']),
            'removed_abstract_count': len(original['abstracts']) - len(withdrawn['abstracts']),
            'observation_cost': 0, 'measurement_cost': 0, 'research_compute_cost': 0,
            'cpu_seconds': time.process_time() - cpu,
            'wall_seconds': time.monotonic() - wall,
            'cost_basis': 'Experimenter knowledge intervention; no forecasts, fitting or measurements. Actual serialization/withdrawal CPU and wall time recorded.',
            'empty_withdrawal_preserves_complete_library': True,
        }
        cpu = time.process_time()
        session = runtime.UnifiedResearchSession(state=state, lab_factory=laboratory)
        result = session.run(max_tasks=1)
        research_cpu = time.process_time() - cpu
        assert len(result['results']) == 3
        save(output / 'dev_knowledge_ablation' / root / 'checkpoint.json.gz', result)
        full_state = read(directory / 'visit03.json.gz')
        full = full_state['results'][2]['research']
        ablated = result['results'][2]['research']
        learned.append({'root': root, 'task': result['tasks'][2], 'withdrawal': withdrawal,
            'report': ablated, 'cpu_seconds': research_cpu,
            'full': {key: full.get(key) for key in (
                'status', 'selected_kind', 'costs', 'used_sources', 'concepts_added')},
            'full_terminal': read(directory / 'evaluation.json')[2]})

    # Both ablated research runs have stopped before any terminal measurements.
    rows = []
    for item in learned:
        report, settings = item.pop('report'), protocol['terminal']
        program = Program.from_dict(report['program'])
        cpu = time.process_time()
        with laboratory(item['task']) as lab:
            errors = []
            for index in range(settings['trajectories']):
                probe = lab.random_probe(item['task']['seed'] + settings['seed_offset'] + 10007 * index,
                    length=settings['length'], name=f'terminal_{index}')
                errors.append(loss(program, lab.acquire(probe).observations, report['fallback'], .5))
            terminal = {'mse': sum(errors) / len(errors), 'trajectories': errors,
                'costs': {**lab.costs, 'compute_cost': settings['length'] * settings['trajectories']},
                'cpu_seconds': time.process_time() - cpu}
        item.update(ablated={key: report.get(key) for key in (
            'status', 'selected_kind', 'costs', 'used_sources', 'concepts_added')},
            ablated_terminal=terminal)
        item['full']['passed'] = item['full_terminal']['mse'] <= protocol['pass_mse']
        item['ablated']['passed'] = terminal['mse'] <= protocol['pass_mse']
        rows.append(item)
    save(output / 'dev_knowledge_ablation.json', {
        'scope': 'Development only; no formal-roster changes or formal causal claim.',
        'intervention': 'Restore the frozen new-arm checkpoint after remove+restore; withdraw only concepts learned by remove and their dependencies, keep unrelated prior/method experience; run replacement stage with identical task and budget.',
        'runtime': protocol['arms']['mechanism_revision']['runtime'],
        'runtime_change': 'Frozen development runtime, with only the KnowledgeLibrary.without preservation fix in memory.py. Empty withdrawal was checked to preserve the complete control library before each intervention.',
        'control': 'Existing complete full-knowledge stage-2 report and terminal scores, not rerun.',
        'rows': rows,
    })
    for row in rows:
        print({key: row[key] for key in ('root', 'full', 'ablated', 'ablated_terminal')}, flush=True)


if __name__ == '__main__':
    main()
