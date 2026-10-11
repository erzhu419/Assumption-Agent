"""Reuse the saved failed development prefix; purchase no new observations."""
import json
from pathlib import Path
from time import perf_counter

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.transfer_condition_research import TransferConditionResearch

ROOT = Path(__file__).resolve().parent


class WholePredicateResearch(TransferConditionResearch):
    def condition(self, program, position, budget, history_index=None):
        return self.predict(program, position, budget, program.size, history_index)


def main():
    path = ROOT / 'development_v3/runs/hegel_v083/67805902695/evaluation_report.json'
    report = json.loads(path.read_text())
    research = report['session']['results'][7]['research']
    rows = [Observation(item['features'], item['target'], tuple(item['history']))
            for batch in research['batches'][:2] for item in batch['observations']]
    plan = next(action['plan'] for action in research['actions']
                if action['action'] == 'test_transfer_condition_predictions')
    calls = {item['hypothesis_id']: Program.from_dict(item['program'])
             for item in plan['hypotheses'] if item['hypothesis_id'] != 'transfer_trial'}
    questions = [(Program.from_dict(condition['predicate']),
                  calls[entry['hypothesis_id']].expanded().history_depth)
                 for entry in plan['transfer_scope']['calls'] for condition in entry['conditions']]
    outputs, measures = {}, {}
    for name, cls in [('whole_predicate', WholePredicateResearch),
                      ('shared_conditions', TransferConditionResearch)]:
        evaluator, budget = cls(()), SearchBudget(1000000)
        evaluator.sync_rows(rows)
        start = perf_counter()
        outputs[name] = [evaluator.phase(predicate, depth, position, budget)
                         for predicate, depth in questions for position in range(len(rows))]
        measures[name] = {'compute_cost': budget.spent, 'wall_seconds': perf_counter() - start,
                          'counts': evaluator.counts}
    assert outputs['whole_predicate'] == outputs['shared_conditions']
    assert measures['shared_conditions']['compute_cost'] < measures['whole_predicate']['compute_cost']
    result = {'source': str(path.relative_to(ROOT)), 'task_position': 7, 'paid_rows': len(rows),
              'questions': len(questions), 'phase_comparisons': len(outputs['whole_predicate']),
              'identical_phase_results': True, 'new_observations': 0, 'evaluation': measures}
    (ROOT / 'paid_condition_evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
