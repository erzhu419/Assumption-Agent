#!/usr/bin/env python3
"""Continue one paid structural question, then gate one untouched confirmation."""
import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from partial_role_paid_scene import load_paid_scene
from run_partial_role_parent import load, save
from run_partial_role_successors import historical_knowledge
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.evaluation import loss, predict
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_continuation import encode_entry, restore_object_state
from hegel_machine.cognition.parallel_prediction import predict_rows
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 90000
SEED = 106029101
FRESH_SEED = SEED + 831272
FRESH_ROWS = 48
TOLERANCE = 1e-7


def prepare(args):
    scene = load_paid_scene(args.root, args.acceptance, args.previous)
    args.output.mkdir(parents=True, exist_ok=False)
    registration = {
        'scope': 'one development local revision; original nine runs unchanged',
        'root': str(args.root.resolve()), 'acceptance': str(args.acceptance.resolve()),
        'previous': str(args.previous.resolve()), 'seed': SEED,
        'compute_limit_including_previous': LIMIT, 'previous_compute': scene['previous_compute'],
        'remaining_compute': LIMIT - scene['previous_compute'],
        'paid_rows': len(scene['paid_rows']), 'old_exposed_rows': len(scene['exposed_rows']),
        'fresh_confirmation_seed': FRESH_SEED, 'maximum_new_rows': FRESH_ROWS,
        'tolerance': TOLERANCE,
        'confirmation_gate': 'changed program, complete same-pool comparison, all-paid MSE, '
            'each retained counterexample and old exposed48 MSE <= original tolerance',
        'failure_gate': 'retain failed or incomplete work; no repeat, new seed or budget increase',
        'historical_knowledge_ref': scene['registration']['historical_knowledge_ref'],
        'sources': scene['source_refs'], 'llm_calls': 0}
    save(args.output / 'registration.json', registration)
    # Freeze current source through the shared incremental writer. Old runtime
    # files provide unchanged blocks only, never the implementation to execute.
    names = [p.relative_to(ROOT) for p in (ROOT / 'src/hegel_machine').rglob('*')
             if p.is_file() and p.suffix in ('.py', '.json')]
    names.extend(Path('scripts') / name for name in (
        'continue_partial_role_structure.py', 'partial_role_paid_scene.py',
        'run_partial_role_parent.py', 'run_partial_role_successors.py',
        'retain_confirmed_structure.py'))
    for name in names:
        target, base = args.output / 'runtime' / name, args.root / 'runtime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_incrementally(target, (ROOT / name).read_text(),
                                 base=base if base.exists() else None)
    return {'registered': str(args.output), 'paid_rows': len(scene['paid_rows']),
            'remaining_compute': registration['remaining_compute']}


def run(output):
    config = load(output / 'registration.json')
    if ROOT != (output / 'runtime').resolve():
        raise ValueError('Run this newly frozen entry; preserve the current and historical checkout')
    if (output / 'receipt.json').exists() or (output / 'failure.json').exists():
        raise ValueError('This unit already ended; retain its result')
    timing = ExecutionTiming(('restore', 'paid_comparison', 'local_revision', 'confirmation', 'save'))
    budget = SearchBudget(LIMIT, spent=config['previous_compute'])
    before = budget.spent
    with timing.phase('restore'):
        scene = load_paid_scene(Path(config['root']), Path(config['acceptance']), Path(config['previous']))
        knowledge, _, _ = historical_knowledge(config['historical_knowledge_ref'])
        agent = UnifiedResearchAgent(ResearchConfig(seed=SEED, minimum_improvement=TOLERANCE,
            max_history=4, compute_budget=LIMIT), library=KnowledgeLibrary.from_dict(knowledge))
        restore_object_state(agent, scene['report'], ('fine',))
        settings = next(x for x in scene['registration']['tasks'] if x['seed'] == SEED)
    channels = tuple(scene['report']['selected_channels'])
    fallback = scene['parent']['fallback']
    prior_rows, exposed_rows, paid_rows = scene['prior_rows'], scene['exposed_rows'], scene['paid_rows']
    prior_scores, exposed_scores = dict(scene['prior_scores']), dict(scene['exposed_scores'])
    actions, confirmation = [], None
    with AnonymousObjectLab(**settings) as laboratory:
        try:
            with python_execution_deadline(1800, activity=True):
                with timing.phase('paid_comparison'):
                    score_start = budget.spent
                    for program in prior_scores:
                        if program not in exposed_scores:
                            if not budget.spend(len(exposed_rows)):
                                raise RuntimeError('Incomplete retained-pool comparison')
                            exposed_scores[program] = loss(program, exposed_rows, fallback, .5)
                    combined = {program: (len(prior_rows) * value + len(exposed_rows) * exposed_scores[program])
                                / len(paid_rows) for program, value in prior_scores.items()}
                    added_scoring = budget.spent - score_start
                    current_loss = combined[scene['parent']['candidate'].program]
                    scene['parent'].update(training_loss=current_loss, validation_loss=current_loss)
                # This includes final old/new frozen predictions. Known paid
                # counterexamples are never replaced by a random success.
                budget.reserve = 2 * FRESH_ROWS + len(paid_rows)
                with timing.phase('local_revision'):
                    result = agent._revise_paid_structure(agent._spec(laboratory, channels),
                        scene['parent'], paid_rows, budget, actions, finalists=scene['pool'],
                        transitions=scene['transitions'], paid_scores=combined)
                budget.release_reserve()
                selected = result['selected']
                selected_program = selected['candidate'].program
                predictions = result.get('selected_predictions')
                errors = ([(value - row.target) ** 2 for row, value in zip(paid_rows, predictions)]
                          if predictions is not None else None)
                if errors is None and result['full_comparison'] and result['status'] == 'improved':
                    if not budget.spend(len(paid_rows)):
                        raise RuntimeError('Incomplete selected residual readout')
                    errors = [(predict(selected_program, row, fallback) - row.target) ** 2 for row in paid_rows]
                targets = scene['counterexample_indices']
                gate = bool(result['full_comparison'] and result['status'] == 'improved' and errors
                    and mean(errors) <= TOLERANCE
                    and all(errors[index] <= TOLERANCE for index in targets)
                    and mean(errors[len(prior_rows):]) <= TOLERANCE)
                with timing.phase('save'):
                    save(output / 'new_claims.json', [encode_entry(item) for item in result['new_claims']])
                    save(output / 'selected_entry.json', encode_entry(selected))
                    score_refs = []
                    for item in result['compared_claims']:
                        old_index = next((i for i, old in enumerate(scene['pool'])
                            if old['candidate'] == item['candidate']), None)
                        ref = (scene['source_refs']['pool'][old_index] if old_index is not None else
                            {'file': 'new_claims.json', 'json_path': [next(i for i, new in
                             enumerate(result['new_claims']) if new['candidate'] == item['candidate'])]})
                        score_refs.append({'candidate_ref': ref, 'candidate_id': item['candidate'].candidate_id,
                                           'loss': item['validation_loss']})
                    save(output / 'selection.json', {
                        'status': result['status'], 'full_comparison': result['full_comparison'],
                        'source_refs': scene['source_refs'], 'new_claims_ref': 'new_claims.json',
                        'selected_entry_ref': 'selected_entry.json', 'actions': actions,
                        'old188_scores_reused': len(scene['pool']), 'additional_exposed48_compute': added_scoring,
                        'scores': score_refs,
                        'selected_paid_errors': errors, 'counterexample_indices': targets,
                        'fresh_confirmation_admitted': gate})
                if gate:
                    with timing.phase('confirmation'):
                        save(output / 'frozen_predictor.json', {
                            'program': selected_program.to_dict(), 'fallback': fallback,
                            'selected_channels': list(channels), 'selected_entry_ref': 'selected_entry.json',
                            'consumed_evidence_ref': 'selection.json', 'status': 'frozen_before_new_confirmation'})
                        if not budget.spend(2 * FRESH_ROWS):
                            raise RuntimeError('Frozen confirmation was not fully funded')
                        probe = laboratory.random_probe(FRESH_SEED, length=FRESH_ROWS,
                                                        name='paid_local_structure_confirmation')
                        batch = laboratory.acquire(probe, channels)
                        save(output / 'confirmation_observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                        new_values = predict_rows(selected_program, batch.observations)
                        old_values = predict_rows(scene['parent']['candidate'].program, batch.observations)
                        new_mse = mean((row.target - v) ** 2 for row, v in zip(batch.observations, new_values))
                        old_mse = mean((row.target - v) ** 2 for row, v in zip(batch.observations, old_values))
                        confirmation = {'rows': FRESH_ROWS, 'incumbent_mse': old_mse, 'selected_mse': new_mse,
                            'selected_predictions': new_values, 'incumbent_predictions': old_values,
                            'threshold': TOLERANCE, 'passed': new_mse <= TOLERANCE}
                receipt = {'status': 'confirmed' if confirmation and confirmation['passed'] else 'unresolved',
                    'selection_status': result['status'], 'full_comparison': result['full_comparison'],
                    'paid_rows': len(paid_rows), 'original188_mse': mean(errors[:len(prior_rows)]) if errors else None,
                    'exposed48_mse': mean(errors[len(prior_rows):]) if errors else None,
                    'paid_mse': selected['validation_loss'] if result['full_comparison'] else None,
                    'counterexample_errors': [errors[i] for i in targets] if errors else None,
                    'new_claims': len(result['new_claims']), 'retained_pool': len(scene['pool']),
                    'additional_exposed48_compute': added_scoring,
                    'fresh_confirmation_admitted': gate, 'confirmation': confirmation,
                    'compute_cost': budget.spent - before, 'cumulative_development_compute': budget.spent,
                    'original_trial_costs': scene['report']['costs'], **laboratory.costs,
                    'llm_calls': 0, 'execution_timing': timing.result(),
                    'scope': 'one existing local reviser; not independent learned-method or ABC gain'}
                save(output / 'receipt.json', receipt)
        except BaseException as error:
            save(output / 'failure.json', {'error': repr(error), 'compute_known': budget.spent - before,
                'cumulative_compute_known': budget.spent, 'compute_upper_bound': LIMIT,
                'cost_scope': 'retain any unfinished nested computation as unknown within limit',
                **laboratory.costs, 'execution_timing': timing.result()})
            raise
    if receipt['status'] == 'confirmed':
        from scripts.retain_confirmed_structure import retain
        try:
            retain(output, output / 'knowledge')
        except BaseException as error:
            # Scientific confirmation stays successful even if its subsequent
            # knowledge export is interrupted; preserve both outcomes.
            save(output / 'knowledge_failure.json', {'error': repr(error),
                'confirmation_ref': str(output / 'receipt.json')})
            raise
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    print(prepare(args) if args.mode == 'prepare' else run(args.output.resolve()))
