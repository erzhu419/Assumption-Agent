import importlib.util
import pickle
import numpy as np

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('batch_solver', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl', 'rb') as incoming:
    s = pickle.load(incoming)[0]['study']
train = s['train']
num = len(train['x'])

def verify(tr, te):
    y, traj = train['y'][tr], train['traj'][tr]
    fit = dict(s, train={'x': train['x'][tr], 'y': y, 'traj': traj,
                         'summary': np.array([y.mean(), y.std(), traj[:, -1].mean(), traj[:, -1].std()])},
               test={'x': train['x'][te]})
    p = mod.solve_batch(fit)
    ey = abs(p['y_hat'] - train['y'][te])
    es = abs(p['state_pred'] - train['traj'][te])
    expected = [p['y_hat'].mean(), p['y_hat'].std(), p['state_pred'][:, -1].mean(), p['state_pred'][:, -1].std()]
    np.testing.assert_allclose(p['diagnostics_vec'], expected, atol=0, rtol=0)
    return [ey.mean(), es.mean(), es.mean(0).max()]

ids = np.arange(num)
results = []
for n in (52, 56):
    metrics = verify(ids[:n], ids[n:])
    print(f'{n}/{num-n} MAEs (response / trajectory / worst time):', metrics)
    results.append(metrics)
rng = np.random.default_rng(772)
for n in (52, 56):
    metrics = []
    for _ in range(12):
        ids = rng.permutation(num)
        metrics.append(verify(ids[:n], ids[n:]))
    print(f'{n}/{num-n} randomized worst MAEs:', np.max(metrics, axis=0))
    results.extend(metrics)
# Hold out both tails of the calibrated radius to probe extrapolation.
z = mod._calibrated_coordinates(s['panel_a'], train['x'])
ids = np.argsort(np.linalg.norm(z, axis=1))
for tr, te in ((ids[:-16], ids[-16:]), (ids[16:], ids[:16])):
    metrics = verify(tr, te)
    print('radius-tail MAEs:', metrics)
    results.append(metrics)
assert np.max(results) <= .038
