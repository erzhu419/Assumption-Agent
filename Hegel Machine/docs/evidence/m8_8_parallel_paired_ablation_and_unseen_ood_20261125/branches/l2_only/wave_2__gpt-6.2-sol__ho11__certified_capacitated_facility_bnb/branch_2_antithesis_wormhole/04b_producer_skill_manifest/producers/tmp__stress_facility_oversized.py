import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('facility_cert', '/holdout_task/facility_cert.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_facility.pkl', 'rb') as stream:
    public = pickle.load(stream)
ref = 195.79165557696302
baseline = solver.solve_and_certify_facilities(public)
i = baseline['open_y'].index(0)
f = public['fixed_cost'].copy()
f[i] = 1e6
# The public incumbent leaves i closed. Any solution opening it now costs
# at least 1e6, so changing its capacity cannot change the public optimum.
for capacity in (public['capacity'][i], 1e6, 1e12, 1e15):
    a = public['capacity'].copy()
    a[i] = capacity
    r = solver.solve_and_certify_facilities(dict(public, fixed_cost=f, capacity=a))
    y = np.asarray(r['open_y']); x = r['flows_X']
    ub = r['upper_bound']; lb = min(z['lower_bound'] for z in r['proof_leaves'])
    print('facility', i, 'capacity', capacity, 'objective', ub, 'gap', ub-lb, 'leaves', len(r['proof_leaves']), flush=True)
    assert y[i] == 0 and abs(ub-ref) < 1e-8
    assert abs(f@y + np.sum(public['cost_matrix']*x)-ub) < 1e-8
    assert x.min() >= 0 and np.max(np.abs(x.sum(0)-public['demand'])) < 1e-9 and np.max(x.sum(1)-a*y) < 1e-9
    assert lb <= ref+1e-10 and ub-lb <= 1e-5 and 1 <= len(r['proof_leaves']) <= 80
