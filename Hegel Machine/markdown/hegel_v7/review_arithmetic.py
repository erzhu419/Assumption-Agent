"""Arithmetic-only check of values quoted in d77 repository evidence.
This does not execute Hegel, fit a model, or reproduce a scientific cohort.
"""
import json
from pathlib import Path

facts = {
    'd1_counts': {'corrected_training_but_held_failed': 16,
                  'full_rank_returned_fit_training_failed': 431,
                  'rank_deficient_unknown': 276},
    'old97': {'old259': 0.06169750106856217, 'new455': 0.06134452617735288},
    'terminal512': {'old259_reported_rounded': 0.073459217,
                    'new455_reported_rounded': 0.079555686},
    'native144validation': {'old259': 0.07948430214236583,
                            'new9': 0.05074486024077334},
    'new7_next_unit': {'native_cap': 500000, 'complete_comparison': 2196379,
                      'two_frozen_reads': 1226, 'stale_reserve': 613,
                      'total': 2698218, 'available_compute': 4204909},
    'python_quote': {'old_score_calls': 80.42005683749449,
                     'old_total_comparison': 137.10068928392138,
                     'old_rows': 609, 'new_rows': 613,
                     'old_pool': 3578, 'new_pool_bound': 3583,
                     'slowest_old_score': 2.6171170349698514,
                     'additional_model_calls_bound': 7},
}
r = {'scope': 'arithmetic_only_not_project_tests_or_scientific_replication',
     'commit': 'd77b2d7d168187220fe6c850f75c5d59ed3a0111',
     'quoted_facts': facts}
r['D1_total'] = sum(facts['d1_counts'].values())
r['old97_relative_mse_reduction'] = 1-facts['old97']['new455']/facts['old97']['old259']
r['terminal512_relative_increase_rounded_inputs'] = (
    facts['terminal512']['new455_reported_rounded']/facts['terminal512']['old259_reported_rounded']-1)
r['old512_collision_mse_floor'] = 3.5/512
r['native144_relative_reduction_local_comparison_only'] = (
    1-facts['native144validation']['new9']/facts['native144validation']['old259'])
b = facts['new7_next_unit']
r['quoted_unit_sum'] = b['native_cap']+b['complete_comparison']+b['two_frozen_reads']+b['stale_reserve']
r['comparison_share_of_compute_quote'] = b['complete_comparison']/b['total']
r['remaining_compute_after_conservative_quote'] = b['available_compute']-b['total']
q = facts['python_quote']
r['python_non_score_call_component'] = q['old_total_comparison']-q['old_score_calls']
r['python_empirical_quote'] = (
    q['old_score_calls']*q['new_rows']/q['old_rows']
    +(q['old_total_comparison']-q['old_score_calls'])*q['new_pool_bound']/q['old_pool']
    +q['additional_model_calls_bound']*q['slowest_old_score']*q['new_rows']/q['old_rows'])
assert r['D1_total'] == 723
assert r['quoted_unit_sum'] == b['total']
assert abs(r['old512_collision_mse_floor']-0.0068359375)<1e-15
assert abs(r['python_empirical_quote']-156.14825344692935)<1e-9
Path(__file__).with_name('arithmetic_checks.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='quoted_facts'}, ensure_ascii=False, indent=2))
