import importlib.util
import pickle
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.solve_batch

solvers = [load('/tmp/baseline_solver.py'), load('/holdout_task/batch_solver.py')]
with open('/holdout_task/public_batch.pkl','rb') as f:
    s = pickle.load(f)[0]['study']
t = s['train']
x = t['x']
rng = np.random.default_rng(1029)
ids = np.arange(len(x))
edge = []
p = np.argsort(np.linalg.norm((x-x.mean(0))/x.std(0), axis=1))
edge.append((p[:56],p[56:]))
for v in x.T:
    p = np.argsort(v)
    edge.extend(((p[12:],p[:12]),(p[:-12],p[-12:])))
for label, splits in [('public', [(ids[:52],ids[52:])]),
                      ('fresh 56/16', [(p[:56],p[56:]) for p in (rng.permutation(ids) for _ in range(20))]),
                      ('edge', edge)]:
    errors = []
    for a,b in splits:
        f = dict(s, train={k:t[k][a] for k in ('x','y','traj')}, test={'x':x[b]})
        e = []
        for solve in solvers:
            out = solve(f)
            e.append([np.mean(abs(out['y_hat']-t['y'][b])), np.mean(abs(out['state_pred']-t['traj'][b])), np.mean(abs(out['state_pred'][:,-1]-t['traj'][b,-1]))])
        errors.append(e)
    e = np.array(errors)
    print(label, 'mean', e.mean(0).round(6).tolist(), 'worst', e.max(0).round(6).tolist())
