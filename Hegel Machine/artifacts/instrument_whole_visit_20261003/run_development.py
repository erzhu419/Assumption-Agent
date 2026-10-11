"""One natural continuation from an existing paid prefix; no terminal evaluation."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_observed_condition_decoding_study import laboratory, load_runtime
from run_knowledge_lineage_study import read, save
from finish_summary import trim


def main():
    protocol = read(BASE / 'successor_regression/protocol.json')
    development = read(BASE / 'development_protocol.json')
    protocol['arms']['new_cumulative']['runtime'] = development['candidate_runtime']
    module = load_runtime(protocol, 'new_cumulative')
    source = ROOT / development['source']
    original = read(source)
    prefix_count = len(original['revisits'])
    session = module.UnifiedResearchSession(state=original, lab_factory=laboratory)
    prefix_costs = session._costs()
    result = {'complete':False, 'source':development['source'],
        'candidate_runtime':development['candidate_runtime'], 'terminal_scores':0,
        'external_model_calls':0, 'before_remaining':session._remaining(),
        'prefix_revisits':prefix_count, 'prefix_costs':prefix_costs, 'visits':[]}
    original_perform = module.UnifiedResearchSession._perform_revisit

    def perform(current, offer, **kwargs):
        position = offer['task_position']
        previous = current.state['results'][position]['research']
        before_actions = len(previous['actions'])
        before_costs = current._costs()
        start = time.process_time()
        executed = original_perform(current, offer, **kwargs)
        elapsed = time.process_time() - start
        after = current.state['results'][position]['research']
        events = [trim(event) for event in after['actions'][before_actions:]]
        entry = {'position':position, 'basis':offer.get('basis'), 'executed':executed,
            'quote':{key:deepcopy(offer.get(key)) for key in
                ('relative_gain','minimum_resources','expected_resource_costs','investment_decision')},
            'cpu_seconds':elapsed,
            'whole_incremental_costs':{key:current._costs()[key]-value for key,value in before_costs.items()},
            'visit_costs':current.state['revisits'][-1]['visit_costs'] if executed else None,
            'before_status':previous['status'], 'after_status':after['status'],
            'program_changed':previous.get('program') != after.get('program'),
            'paid_events':events}
        result['visits'].append(entry)
        save(BASE / 'development_result.json', result)
        print({'position':position, 'executed':executed,
               'costs':entry['whole_incremental_costs']}, flush=True)
        return executed

    module.UnifiedResearchSession._perform_revisit = perform
    cpu, wall = time.process_time(), time.monotonic()
    state = session.run()
    result.update(complete=True, status=state['status'],
        remaining_resources=session._remaining(), cpu_seconds=time.process_time()-cpu,
        wall_seconds=time.monotonic()-wall,
        suffix_costs={key:session._costs()[key]-value for key,value in prefix_costs.items()})
    prior_dir = ROOT / 'artifacts/downstream_research_value_20261003/successor_regression/runs/new_cumulative/root07'
    old = read(prior_dir / 'research_evidence.json.gz')
    old_status = read(prior_dir / 'learning_status.json')
    result['reused_old_suffix'] = {
        'version':'0.168.0', 'source':str(prior_dir.relative_to(ROOT)),
        'visit_positions':[v['task_position'] for v in old['revisits'][prefix_count:]],
        'suffix_costs':{key:old['costs'][key]-value for key,value in prefix_costs.items()},
        'suffix_research_cpu_seconds':old_status['research_cpu_seconds']-original['study']['cpu_seconds'],
        'paid_visit_costs':[v['visit_costs'] for v in old['revisits'][prefix_count:]],
        'terminal_not_read':True,
    }
    result['cost_scope'] = 'Whole incremental visit costs include the execution and the scheduler screening after it; the separate visit_costs field is execution only. Suffix totals include initial queue refresh too.'
    save(BASE / 'development_result.json', result)
    (BASE / 'development_final.json.gz').write_bytes(gzip.compress(
        json.dumps(state, separators=(',', ':')).encode(), compresslevel=1))
    print({'complete':True, 'new_revisits':len(result['visits']),
        'suffix_costs':result['suffix_costs'], 'terminal_scores':0}, flush=True)


if __name__ == '__main__':
    main()
