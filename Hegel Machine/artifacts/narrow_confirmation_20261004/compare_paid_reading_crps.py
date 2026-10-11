"""Score frozen reading forecasts on already-paid subsequent probe batches."""
import gzip
import json
from pathlib import Path
from statistics import mean
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.observed_condition_experiments import _local_residuals
from run_observed_condition_decoding_study import laboratory

ROOT = Path(__file__).parent

def crps(knots, observed):
    """Exact integral for the planner's equal-mass adjacent uniform cells."""
    knots = sorted(knots)
    if len(knots) == 1:
        return abs(knots[0] - observed)
    score = max(0., knots[0] - observed, observed - knots[-1])
    n = len(knots) - 1
    for index, (left, right) in enumerate(zip(knots, knots[1:])):
        if right <= left:
            continue  # Atoms change CDF values, but occupy no integration width.
        for lo, hi in ((left, min(right, observed)), (max(left, observed), right)):
            if hi <= lo:
                continue
            offset = 0. if hi <= observed else 1.
            flo = (index + (lo - left) / (right - left)) / n - offset
            fhi = (index + (hi - left) / (right - left)) / n - offset
            score += (hi - lo) * (flo * flo + flo * fhi + fhi * fhi) / 3.
    return score

with gzip.open(ROOT / 'root09_p10_paid_case.json.gz', 'rt') as stream:
    case = json.load(stream)
spec = laboratory(case['task']).spec_for(case['channels'])
controls = tuple(spec.feature_bounds)
all_batches = case['paid_batches_before_first_confirmation'] + case['actual_confirmation_batches']
rows = []
starts = {}
for batch in all_batches:
    starts[batch['probe']['probe_id']] = len(rows)
    rows.extend(Observation(row['features'], row['target'], tuple(row['history']))
                for row in batch['observations'])
plans = [case['candidate']['instrument_condition_experiment']['plan'],
         *(event for event in case['actions'] if event['action'] == 'plan_condition_confirmation')]
results = []
for stage, plan in enumerate(plans):
    probe_id = plan.get('probe_id', plan.get('probe', {}).get('probe_id'))
    batch = next(batch for batch in all_batches if batch['probe']['probe_id'] == probe_id)
    before = starts[probe_id]
    assert plan['paid_reading_rows'] == before
    paid = rows[:before]
    field = plan['condition_field']
    fitted = Program.from_dict(plan['reading_mean_program'])
    residuals = [row.features[field] - fitted.predict(row) for row in paid]
    point = plan['fixed_control_background']
    _, local, _ = _local_residuals(paid, residuals, controls, spec.feature_bounds,
                                  point, SearchBudget(100000))
    predicted_mean = fitted.predict(Observation(point, 0.))
    local_cdf = [predicted_mean + value for value in local]
    shared_cdf = [predicted_mean + value for value in residuals]
    actual = [row['features'][field] for row in batch['observations']]
    local_score = mean(crps(local_cdf, value) for value in actual)
    shared_score = mean(crps(shared_cdf, value) for value in actual)
    results.append({'stage': 'pre_pair_paid_revision' if stage == 0 else 'pair_confirmation',
        'probe_id': probe_id, 'paid_training_rows': before, 'heldout_rows': len(actual),
        'local_rows': len(local), 'local_crps': local_score, 'shared_crps': shared_score,
        'shared_minus_local': shared_score - local_score,
        'all_forecast_readings_precede_holdout': True,
        'would_choose_by_this_paid_holdout': 'shared' if shared_score < local_score else 'local'})
out = {'scope': 'Read-only chronological evaluation; each saved mean and both residual CDFs use only rows paid before the scored batch. No terminal data or acquisitions.',
       'distribution': 'Same adjacent equal-mass residual CDF as the pair planner, with tied knots retaining atoms.',
       'results': results,
       'interpretation': 'The first scored batch was already paid before the first pair plan, so its reading-only comparison is available for model choice then. Later batches can inform only subsequent plans.',
       'limits': ['One real target case; does not establish general superiority of a shared residual distribution.',
                  'A smaller reading CRPS is not direct proof of a better rare-strip probability or final response accuracy.']}
(ROOT / 'root09_paid_reading_crps.json').write_text(json.dumps(out, indent=2) + '\n')
for row in results:
    print(json.dumps(row))
