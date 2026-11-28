import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
train = study['train']
x, y, traj = (train[k] for k in ('x', 'y', 'traj'))
n = len(x)
all_rows = np.arange(n)
z = mod._calibrate(study['panel_a'], x)
order = np.argsort(np.linalg.norm(z, axis=1))

def check(out, count):
    a, b, d = (out[k] for k in ('y_hat', 'state_pred', 'diagnostics_vec'))
    assert a.shape == (count,) and b.shape == (count, len(study['eval_times'])) and d.shape == (4,)
    assert all(np.isfinite(v).all() for v in (a, b, d))
    end = b[:, -1]
    np.testing.assert_allclose(d, [a.mean(), a.std(), end.mean(), end.std()])
    return a, b

check(mod.solve_batch(study), len(study['test']['x']))

def score(hold):
    keep = np.setdiff1d(all_rows, hold)
    yy, tt = y[keep], traj[keep]
    fit = dict(x=x[keep], y=yy, traj=tt,
               summary=np.array([yy.mean(), yy.std(), tt[:, -1].mean(), tt[:, -1].std()]))
    a, b = check(mod.solve_batch(dict(study, train=fit, test={'x': x[hold]})), len(hold))
    e, g = abs(a - y[hold]), abs(b - traj[hold])
    return [e.mean(), g.mean(), g[:, -1].mean(), g.mean(0).max()]

print('slice: label MAE, trajectory MAE, terminal MAE, worst timestamp MAE')
for name, indices in [('52/20', all_rows[52:]), ('56/16', all_rows[56:]),
                      ('largest calibrated norm', order[-16:]),
                      ('smallest calibrated norm', order[:16])]:
    result = score(indices)
    print(name, np.round(result, 6))
    assert max(result) < .038
rng = np.random.default_rng(914)
res = np.array([score(k) for _ in range(6) for k in np.array_split(rng.permutation(n), 4)])
print('24 held-out folds, worst', np.round(res.max(0), 6))
assert res.max() < .038
