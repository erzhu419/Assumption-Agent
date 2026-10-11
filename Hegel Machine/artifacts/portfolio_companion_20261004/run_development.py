"""Paired paid-prefix study observing actual selected-companion execution."""
from copy import deepcopy
import importlib.util
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


def pending_view(offers):
    return [{'position':offer['task_position'],'source_position':offer.get('source_task_position'),
        'basis':offer.get('basis'),'minimum_resources':deepcopy(offer.get('minimum_resources')),
        'expected_resource_costs':deepcopy(offer.get('expected_resource_costs')),
        'expected_loss_reduction':offer.get('expected_loss_reduction'),
        'instrument_decision':deepcopy(offer.get('candidate',{}).get('instrument_condition_experiment',{}).get('decision')),
        'investment_decision':deepcopy(offer.get('investment_decision')),
        'draft_ids':deepcopy(offer.get('candidate',{}).get('method_opportunity',{}).get('draft_ids'))}
        for offer in offers]


def learn(protocol,arm,root):
    visits=[]
    def load(*args):
        module=original_load(*args)
        parent=module.UnifiedResearchSession
        class ObserveCompanion(parent):
            def _perform_revisit(self,offer,**kwargs):
                decision=next((event for event in reversed(self.state['scheduler_events'])
                    if event['action']=='choose_research_portfolio'),None)
                record={'before_revisits':len(self.state['revisits']),
                    'selected':pending_view([offer])[0],
                    'allocation_limits':deepcopy(kwargs.get('allocation_limits')),
                    'portfolio_decision':deepcopy(decision),
                    'remaining_before':self._remaining(),
                    'pending_before':pending_view(self.state['pending_revisits'])}
                result=super()._perform_revisit(offer,**kwargs)
                record.update(executed=bool(result),remaining_after=self._remaining(),
                    pending_after=pending_view(self.state['pending_revisits']))
                if result:
                    record['actual_visit_costs']=deepcopy(self.state['revisits'][-1]['visit_costs'])
                visits.append(record)
                return result
        module.UnifiedResearchSession=ObserveCompanion
        return module
    runner.load_runtime=load
    original_learn(protocol,arm,root)
    runner.save(BASE/'runs'/arm/root/'companion_execution.json',{'complete':True,
        'root':root,'arm':arm,'actual_visits':visits})

runner.learn=learn
runner.__file__=str(ENTRY)
if __name__=='__main__':
    runner.main()
