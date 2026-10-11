"""Reuse paired paid-prefix study; observe actual method feedback before pricing."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path

BASE=Path(__file__).resolve().parent
os.environ['HEGEL_PORTFOLIO_OUTPUT']=str(BASE)
spec=importlib.util.spec_from_file_location('portfolio_resume_runner',
    BASE.parent/'research_portfolio_20261003'/'run_development.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_learn=runner.learn
original_load=runner.load_runtime


def learn(protocol,arm,root):
    outcomes=[]
    def load(*args):
        module=original_load(*args)
        parent=module.UnifiedResearchSession
        class ObserveMethodFeedback(parent):
            def _offer_new_knowledge(self,position,*args,**kwargs):
                visits=self.state['revisits']
                if visits and visits[-1]['task_position']==position:
                    visit=visits[-1]
                    outcome=visit.get('method_visit_outcome')
                    if outcome is not None:
                        context=outcome.get('method_context')
                        outcomes.append({'after_revisits':len(visits),'position':position,
                            'source_task_position':visit['source_task_position'],
                            'outcome':runner.compact(deepcopy(outcome)),
                            'method_experience':runner.compact(deepcopy(
                                self.library.action_experience.get(context,{}))),
                            'response_experience':{item['method_context']:runner.compact(deepcopy(
                                self.library.action_experience.get(item['method_context'],{})))
                                for item in outcome.get('response_confirmations',[])}})
                return super()._offer_new_knowledge(position,*args,**kwargs)
        module.UnifiedResearchSession=ObserveMethodFeedback
        return module
    runner.load_runtime=load
    original_learn(protocol,arm,root)
    runner.save(BASE/'runs'/arm/root/'method_feedback.json',{'complete':True,
        'root':root,'arm':arm,'outcomes_before_downstream_quotation':outcomes})

runner.learn=learn
# Preserve the generic runner's worker entry points and output configuration.
runner.__file__=str(Path(__file__).resolve())
if __name__=='__main__':
    runner.main()
