"""Three paid prefixes; same coherent planner and equal new procurement caps."""
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
FIXTURE = ROOT / 'artifacts/coherent_confirmation_20261003/paid_confirmation_cases.json.gz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime)
    if not runtime.is_absolute():
        runtime = ROOT / runtime
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from hegel_machine.cognition.condition_confirmation import acquire_condition_confirmation
    from hegel_machine.cognition.evaluation import predict
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
    from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
    from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
    from hegel_machine.cognition.unified_research import ResearchConfig, UnifiedResearchAgent
    from run_observed_condition_decoding_study import laboratory
    from investment_ablation import acquire_fixed_pair_confirmation

    with gzip.open(FIXTURE, 'rt') as stream:
        fixture = json.load(stream)
    caps = {'observation_cost':96, 'measurement_cost':96, 'compute_cost':90000}
    result = {'complete':False, 'runtime':str(runtime.relative_to(ROOT)),
        'fixture':str(FIXTURE.relative_to(ROOT)), 'caps_per_arm':caps, 'cases':[],
        'scope':'Three known paid prefixes; common reconstructed state with a new equal procurement allowance, not exact historical continuation.',
        'posthoc_scope':'Per-batch frozen predictions below are diagnostic only. The actual helper and returned-batch caller predictions are charged by their production budget calls.',
        'external_model_calls':0, 'terminal_scores':0}
    for case in fixture['cases']:
        channels = tuple(case['channels'])
        paid_batches = case['paid_batches_before_confirmation']
        length = len(case['recorded_fresh_batch']['observations'])
        old_program, new_program = map(Program.from_dict, (case['baseline_program'],case['program']))
        rows = []
        for batch in paid_batches:
            if not set(channels) <= set(batch['channels']):
                continue
            history = []
            for value in batch['observations']:
                rows.append(Observation(dict(value['features']),value['target'],
                    tuple(deepcopy(value.get('history',history)))))
                history.append({**value['features'],'__target__':value['target']})
        for arm in ('prior_investment','pair_investment'):
            cpu = time.process_time()
            budget = SearchBudget(caps['compute_cost'])
            selected = {'candidate':Candidate('paid_revision',new_program,'paid_revision'),
                'fallback':case['fallback'],'channels':channels,
                'research_program':deepcopy(case['continuation_candidate'].get('research_program'))}
            baseline = {'candidate':Candidate('confirmation_baseline',old_program,'explanation_before_research'),
                'fallback':case['baseline_fallback'],'channels':channels}
            actions, purchased = [], []
            all_rows = list(rows)
            with laboratory(case['task']) as lab:
                lab.restore([{**batch['probe'],'channels':batch['channels']} for batch in paid_batches])
                paid_costs = dict(lab.costs)
                spec = lab.spec_for(channels)
                rates = {channel.name:channel.cost_per_row for channel in lab.channels}
                rebuilding = SearchBudget(15000)
                state = assess_observed_condition_uncertainty(spec,rows,selected,rebuilding,
                    previous=case['continuation_candidate'].get('condition_uncertainty'),
                    tolerance=case['config']['minimum_improvement'])
                budget.spend(rebuilding.spent)
                if state is not None:
                    selected['condition_uncertainty'] = state
                planning = SearchBudget(15000)
                plan = plan_observed_condition_probe(spec,rows,selected,planning,
                    serial=len(paid_batches),length=length,
                    tolerance=case['config']['minimum_improvement'],confirmation_baseline=baseline)
                budget.spend(planning.spent)

                def remaining():
                    return {'observation_remaining':caps['observation_cost']-(lab.costs['observation_cost']-paid_costs['observation_cost']),
                        'measurement_remaining':caps['measurement_cost']-(lab.costs['measurement_cost']-paid_costs['measurement_cost']),
                        'compute_remaining':budget.remaining}

                def acquire(requested_channels,purpose,*,probe):
                    rate = sum(rates[name] for name in requested_channels)
                    available = remaining()
                    if probe.cost>available['observation_remaining'] or probe.cost*rate>available['measurement_remaining']:
                        return None
                    batch = lab.acquire(probe,requested_channels)
                    purchased.append((probe,batch))
                    all_rows.extend(batch.observations)
                    actions.append({'action':'acquire','purpose':purpose,'probe_id':probe.probe_id,
                        'observation_cost':batch.observation_cost,'measurement_cost':batch.measurement_cost})
                    return batch.observations

                def rows_for(requested_channels):
                    return all_rows

                procurement_start = budget.spent
                if plan is None:
                    agent = UnifiedResearchAgent(replace(ResearchConfig(**case['config']),seed=case['task']['seed']))
                    probe = agent._probe(spec,len(paid_batches),length,(),budget,actions)
                    validation = acquire(channels,'confirm_continuation_program',probe=probe)
                else:
                    consumer = acquire_fixed_pair_confirmation if arm=='prior_investment' else acquire_condition_confirmation
                    validation = consumer(spec,selected,plan,budget,actions,acquire=acquire,
                        remaining=remaining,rows_for=rows_for,channels=channels,
                        measurement_rate=sum(rates[name] for name in channels),
                        tolerance=case['config']['minimum_improvement'])
                procurement_compute = budget.spent-procurement_start
                caller_compute = 0
                caller_funded = validation is not None and budget.spend(2*len(validation))
                caller_result = None
                if caller_funded:
                    caller_compute = 2*len(validation)
                    caller_old = [predict(old_program,row,case['baseline_fallback']) for row in validation]
                    caller_new = [predict(new_program,row,case['fallback']) for row in validation]
                    caller_result = {
                        'old_loss':sum((value-row.target)**2 for value,row in zip(caller_old,validation))/len(validation),
                        'new_loss':sum((value-row.target)**2 for value,row in zip(caller_new,validation))/len(validation)}
                elapsed_cpu = time.process_time()-cpu
                batches = []
                for probe,batch in purchased:
                    old = [predict(old_program,row,case['baseline_fallback']) for row in batch.observations]
                    new = [predict(new_program,row,case['fallback']) for row in batch.observations]
                    deltas = [(a-b)**2 for a,b in zip(old,new)]
                    batches.append({'probe':asdict(probe),'fresh_rows':[asdict(row) for row in batch.observations],
                        'returned_to_caller':validation is batch.observations,
                        'old_predictions':old,'new_predictions':new,
                        'old_loss':sum((a-row.target)**2 for a,row in zip(old,batch.observations))/len(old),
                        'new_loss':sum((a-row.target)**2 for a,row in zip(new,batch.observations))/len(new),
                        'mean_prediction_disagreement':sum(deltas)/len(deltas),
                        'discriminating_rows':sum(value>case['config']['minimum_improvement'] for value in deltas)})
                costs = {key:lab.costs[key]-paid_costs[key] for key in ('observation_cost','measurement_cost')}
                costs['compute_cost'] = budget.spent
                entry = {'root':case['root'],'position':case['task_position'],'arm':arm,
                    'caps':caps,'saved_historical_confirmation_quote':case['continuation_candidate']['instrument_condition_experiment']['confirmation_compute_quote'],
                    'costs':costs,'cpu_seconds':elapsed_cpu,
                    'condition_state_reconstruction_compute':rebuilding.spent,'planner_compute':planning.spent,
                    'procurement_compute':procurement_compute,'caller_prediction_compute':caller_compute,
                    'caller_prediction_funded':caller_funded,'caller_result':caller_result,'remaining':remaining(),
                    'plan':{key:value for key,value in (plan or {}).items() if key not in ('probe','reading_mean_program','competing_hypotheses')},
                    'actions':actions,'batches':batches,
                    'acquired_batches':len(batches),'discriminating_rows':sum(batch['discriminating_rows'] for batch in batches),
                    'returned_fresh_rows':len(validation) if validation is not None else 0}
                result['cases'].append(entry)
                (BASE/'development_result.json').write_text(json.dumps(result,indent=2)+'\n')
                print({key:entry[key] for key in ('root','position','arm','costs','acquired_batches','discriminating_rows','returned_fresh_rows')},flush=True)
    result['complete'] = True
    (BASE/'development_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
