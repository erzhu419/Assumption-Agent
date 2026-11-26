import pickle
import sys
import numpy as np
sys.path.insert(0, '/tmp')
from test_proof_box import check

# Keep the measured X and y; move box constraints and support prices.
cases = pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl', 'rb'))
results = []
for i, case in enumerate(cases):
    p = case['problem']
    results.append(check(p, case['gt']['exact_objective']))
    peak = np.max(np.abs(case['gt']['exact_weights']))
    for M in (0.35 * peak, peak * (1 - 1e-6), peak * (1 + 1e-6), p['M']):
        for factor in (0.55, 0.95, 1.05, 1.45):
            results.append(check(dict(p, M=M, lam0=p['lam0'] * factor)))
print('public and boundary slices:', len(results), 'worst gap:', max(r[1] for r in results), 'max leaves:', max(r[0] for r in results), flush=True)

# The stated threshold permits an uncharged nonzero value, while the
# output contract requires such coordinates to be returned as zero.
x, y, l2, l0 = 1e8, .9, .14, .22
b = x*y/(x*x+l2)
f = .5*(x*b-y)**2 + .5*l2*b*b
assert 0 < b < 1e-8 and f < 1e-5
print('threshold obstruction (embed in 19 columns):', b, 'feasible objective:', f, 'minimum objective with inactive weights set to zero:', min(.5*y*y,l0), flush=True)
