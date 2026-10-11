"""Two actual paid revisions, equal funded fresh batches, no fitting or terminal scores."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
FIXTURE = ROOT / 'artifacts/instrument_whole_visit_20261003/paid_confirmation_root07.json.gz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True, help='Reviewed runtime/src directory, relative to project root or absolute.')
    args = parser.parse_args()
    runtime = Path(args.runtime)
    if not runtime.is_absolute():
        runtime = ROOT / runtime
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from hegel_machine.cognition.evaluation import predict
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
    from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
    from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
    from hegel_machine.cognition.unified_research import ResearchConfig, UnifiedResearchAgent
    from run_observed_condition_decoding_study import laboratory
    from confirmation_ablation import prior_confirmation

    with gzip.open(FIXTURE, 'rt') as stream:
        fixture = json.load(stream)
    result = {'complete':False, 'runtime':str(runtime.relative_to(ROOT)),
        'source_fixture':str(FIXTURE.relative_to(ROOT)), 'cases':[],
        'scope':'Known paid revisions, reconstructed uncertainty and one funded fresh batch; no full historical state recovery or terminal evaluation.',
        'budget_basis':'Each fixture explicitly quotes 30096 confirmation compute;24 observation and24 measurement equal its actual recorded fresh batch. Shared paid-state reconstruction and planning are charged equally within these new caps.',
        'missing_history':'Intermediate condition_state and compute ledger were not saved. Reconstruct only from paid prefix and old research recipe; do not substitute final knowledge or claim exact original-stage replay.',
        'external_model_calls':0, 'terminal_scores':0}
    for case in fixture['cases']:
        position = case['task_position']
        channels = tuple(case['channels'])
        paid_batches = case['paid_batches_before_confirmation']
        count = len(case['recorded_fresh_batch']['observations'])
        compute_cap = case['continuation_candidate']['instrument_condition_experiment']['confirmation_compute_quote']
        old_program, new_program = map(Program.from_dict,
            (case['baseline_program'], case['program']))
        rows = []
        for batch in paid_batches:
            if not set(channels) <= set(batch['channels']):
                continue
            history = []
            for value in batch['observations']:
                past = value.get('history', history)
                rows.append(Observation(dict(value['features']), value['target'], tuple(deepcopy(past))))
                history.append({**value['features'], '__target__':value['target']})
        for arm in ('prior_confirmation', 'paired_confirmation'):
            cpu = time.process_time()
            budget = SearchBudget(compute_cap)
            selected = {'candidate':Candidate('paid_revision', new_program, 'paid_revision'),
                'fallback':case['fallback'], 'channels':channels,
                'research_program':deepcopy(case['continuation_candidate'].get('research_program'))}
            baseline = {'candidate':Candidate('confirmation_baseline', old_program, 'explanation_before_research'),
                'fallback':case['baseline_fallback'], 'channels':channels}
            with laboratory(case['task']) as lab:
                lab.restore([{**batch['probe'], 'channels':batch['channels']} for batch in paid_batches])
                paid_costs = dict(lab.costs)
                spec = lab.spec_for(channels)
                rebuilding = SearchBudget(min(15000, max(0, budget.remaining-15000-2*count)))
                state = assess_observed_condition_uncertainty(spec, rows, selected, rebuilding,
                    previous=case['continuation_candidate'].get('condition_uncertainty'),
                    tolerance=case['config']['minimum_improvement'])
                reconstruction_compute = rebuilding.spent
                budget.spend(reconstruction_compute)
                if state is not None:
                    selected['condition_uncertainty'] = state
                planner_cap = 15000 if arm=='paired_confirmation' else max(0,15000-reconstruction_compute)
                planning = SearchBudget(min(planner_cap, max(0,budget.remaining-2*count)))
                options = dict(serial=len(paid_batches), length=count,
                    tolerance=case['config']['minimum_improvement'], confirmation_baseline=baseline)
                plan = (prior_confirmation(plan_observed_condition_probe, spec, rows, selected, planning, **options)
                    if arm=='prior_confirmation' else plan_observed_condition_probe(spec, rows, selected, planning, **options))
                budget.spend(planning.spent)
                if plan is not None and plan['expected_disagreement'] <= case['config']['minimum_improvement']:
                    plan = None
                actions = []
                if plan is None:
                    config = replace(ResearchConfig(**case['config']), seed=case['task']['seed'])
                    agent = UnifiedResearchAgent(config)
                    probe = agent._probe(spec, len(paid_batches), count, (), budget, actions)
                else:
                    probe = plan['probe']
                fresh = lab.acquire(probe, channels)
                if not budget.spend(2*len(fresh.observations)):
                    raise RuntimeError('The fixed confirmation cap cannot fund the paired predictions.')
                old = [predict(old_program, row, case['baseline_fallback']) for row in fresh.observations]
                new = [predict(new_program, row, case['fallback']) for row in fresh.observations]
                targets = [row.target for row in fresh.observations]
                delta = [(a-b)**2 for a,b in zip(old,new)]
                costs = {key:lab.costs[key]-paid_costs[key] for key in ('observation_cost','measurement_cost')}
                costs['compute_cost'] = budget.spent
                entry = {'position':position, 'arm':arm,
                    'caps':{'observation_cost':count, 'measurement_cost':count, 'compute_cost':compute_cap},
                    'costs':costs, 'cpu_seconds':time.process_time()-cpu,
                    'condition_state_reconstruction_compute':reconstruction_compute,
                    'planner_compute':planning.spent,
                    'planner_cap':planner_cap,
                    'planning_compute_including_reconstruction':planning.spent+reconstruction_compute,
                    'reconstructed_condition':{key:state.get(key) for key in
                        ('status','original_target_loss','expected_prediction_ambiguity','conditions')} if state else None,
                    'plan':{key:deepcopy(value) for key,value in (plan or {}).items()
                        if key not in ('probe','reading_mean_program','competing_hypotheses','confirmation_pair')},
                    'pair_planned':bool(plan and plan.get('confirmation_pair')),
                    'probe':asdict(probe), 'fresh_rows':[asdict(row) for row in fresh.observations],
                    'old_predictions':old, 'new_predictions':new,
                    'old_loss':sum((a-y)**2 for a,y in zip(old,targets))/len(targets),
                    'new_loss':sum((a-y)**2 for a,y in zip(new,targets))/len(targets),
                    'mean_prediction_disagreement':sum(delta)/len(delta),
                    'max_prediction_disagreement':max(delta),
                    'discriminating_rows':sum(value>case['config']['minimum_improvement'] for value in delta),
                    'funded_single_batch':len(fresh.observations)==count}
                result['cases'].append(entry)
                (BASE/'development_result.json').write_text(json.dumps(result, indent=2)+'\n')
                print({key:entry[key] for key in ('position','arm','costs','old_loss','new_loss','discriminating_rows')}, flush=True)
    result['complete'] = True
    (BASE/'development_result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
