"""Capture actual frozen-pair plans and fresh batches during ordinary study runs."""
from copy import deepcopy
import gzip
import importlib.util
import json
import os
from pathlib import Path

ENTRY=Path(__file__).resolve()
BASE=Path(os.environ.get('HEGEL_PORTFOLIO_OUTPUT',str(ENTRY.parent)))
os.environ['HEGEL_PORTFOLIO_OUTPUT']=str(BASE)
spec=importlib.util.spec_from_file_location('portfolio_resume_runner',
    ENTRY.parent.parent/'research_portfolio_20261003'/'run_development.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_learn=runner.learn
original_load=runner.load_runtime


def learn(protocol,arm,root):
    visits=[]
    def load(*args):
        module=original_load(*args)
        parent=module.UnifiedResearchSession
        class ObserveConfirmation(parent):
            def _perform_revisit(self,offer,**kwargs):
                position=offer['task_position']
                before=self.state['results'][position]['research']
                action_start=len(before['actions'])
                result=super()._perform_revisit(offer,**kwargs)
                if not result:
                    return result
                report=self.state['results'][position]['research']
                actions=report['actions'][action_start:]
                plans=[deepcopy(event) for event in actions if event['action']=='plan_condition_confirmation']
                allocations=[deepcopy(event) for event in actions if event['action']=='allocate_pair_confirmation']
                comparisons=[deepcopy(event) for event in actions if event['action'] in (
                    'compare_method_followup_on_fresh_response','compare_successor_method_on_fresh_response')]
                probe_ids={event['probe_id'] for event in plans+comparisons if event.get('probe_id')}
                probe_ids.update(identifier for event in allocations for identifier in event['probe_ids'])
                batches={batch['probe']['probe_id']:deepcopy(batch)
                    for batch in report['batches'] if batch['probe']['probe_id'] in probe_ids}
                if plans or allocations or comparisons:
                    visits.append({'after_revisits':len(self.state['revisits']),'position':position,
                        'basis':offer.get('basis'),'allocation_limits':deepcopy(kwargs.get('allocation_limits')),
                        'visit_costs':deepcopy(report['visit_costs']),'plans':plans,
                        'pair_allocations':allocations,'caller_fresh_comparisons':comparisons,
                        'fresh_batches_by_probe':batches})
                return result
        module.UnifiedResearchSession=ObserveConfirmation
        return module
    runner.load_runtime=load
    original_learn(protocol,arm,root)
    payload={'complete':True,'root':root,'arm':arm,
        'scope':'Raw plans and already paid fresh batches; no extra agent computation, observations, fitting or scoring performed by this recorder.',
        'visits':visits}
    path=BASE/'runs'/arm/root/'pair_confirmation.json.gz'
    path.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),compresslevel=1))

runner.learn=learn
runner.__file__=str(ENTRY)
if __name__=='__main__':
    runner.main()
