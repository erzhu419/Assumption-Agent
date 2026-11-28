import importlib.util, pickle, numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('model', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

m = load('/holdout_task/batch_solver.py'); b = load('/tmp/quadratic_baseline.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']; tr = s['train']; x = tr['x']; n = len(x)
rng = np.random.default_rng(34829)
splits = {'fixed': [np.arange(n)], 'random':[rng.permutation(n) for _ in range(20)], 'tail':[np.argsort(sign*x[:,i]) for i in range(4) for sign in (-1,1)]}
for name, perms in splits.items():
    errors = []
    for order in perms:
        i, j = order[:52], order[52:]
        f = dict(s,train={**tr,**{k:tr[k][i] for k in ('x','y','traj')}},test={'x':x[j]})
        y = tr['y'][j]; t = tr['traj'][j]
        p, g, vt = m._components(f)
        score = lambda yh, th: [np.mean(abs(yh-y)),np.mean(abs(th-t)),np.max(abs(th-t).mean(0))]
        base = b.solve_batch(f)
        errors.append([score(base['y_hat'],base['state_pred'])]+[score((w*p+(1-w)*g)[:,0],((q*p+(1-q)*g)[:,1:])@vt) for w,q in [(1,1),(0,0),(.65,.4),(.8,.65),(.4,.3)]])
    errors = np.array(errors)
    print(name, 'rows=quadratic, poly, smooth, candidate, more_poly, more_smooth; cols=readout, traj, worst_timestamp')
    print('mean', np.round(errors.mean(0), 5)); print('worst',np.round(errors.max(0),5))
    if name == 'fixed': assert errors[0,3,0] <= .038
