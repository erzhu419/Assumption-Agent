import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('facility_cert', '/holdout_task/facility_cert.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_facility.pkl', 'rb') as stream:
    public = pickle.load(stream)
ref = 195.79165557696302
f, a, d, c = (public[k] for k in ('fixed_cost', 'capacity', 'demand', 'cost_matrix'))
m = len(f)
bits = (np.arange(1 << m)[:, None] >> np.arange(m)) & 1
for exp in (0, -36, -48, 36, 48):
    for expensive in (False, True):
        cost = c.copy()
        if expensive:
            cost[7, 13] = 1e12
        p = dict(public, capacity=np.ldexp(a, exp), demand=np.ldexp(d, exp), cost_matrix=np.ldexp(cost, -exp))
        r = solver.solve_and_certify_facilities(p)
        y = np.asarray(r['open_y']); x = np.ldexp(r['flows_X'], -exp)
        primal = max(np.abs(x.sum(0)-d).max(), (x.sum(1)-a*y).max(), -x.min())
        ub, leaves = r['upper_bound'], r['proof_leaves']
        lb = min(leaf['lower_bound'] for leaf in leaves)
        assert abs(ub-ref) < 1e-8 and abs(f@y + np.sum(cost*x)-ub) < 1e-8
        assert primal < 1e-9 and lb <= ref+1e-10 and ub-lb <= 1e-5
        assert 1 <= len(leaves) <= 80
        cover = np.zeros(len(bits), dtype=int)
        for leaf in leaves:
            cover += (bits[:, leaf['fixed_closed']]==0).all(1) & (bits[:, leaf['fixed_open']]==1).all(1)
        assert (cover == 1).all()
        print('exponent', exp, 'expensive arc', expensive, 'objective', ub, 'primal error', primal, 'certificate gap', ub-lb, flush=True)
