import importlib.util
import pickle
import numpy as np
path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', path + 'batch_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(path + 'public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
p = study['train']; x = p['x']; a, _, _ = mod._nodes(study, x)
idx = np.arange(len(x)); rng = np.random.default_rng(425)
splits = [('52/20', idx[:52], idx[52:]), ('56/16', idx[:56], idx[56:])]
for label, score in [('high-radius', np.linalg.norm(a, axis=1)), ('high-coordinate', np.max(abs(a), axis=1))]:
    o = np.argsort(score); splits.append((label, o[:56], o[56:]))
o = np.argsort(np.sum(abs(a) > .35, axis=1)); splits.append(('low-active', o[16:], o[:16]))
for _ in range(30):
    o = rng.permutation(idx); splits.append(('random', o[:52], o[52:]))

def verify(out, n):
    y, state = out['y_hat'], out['state_pred']
    diag = out['diagnostics_vec']; end = state[:, -1]
    for value, shape in [(y, (n,)), (state, (n, len(study['eval_times']))), (diag, (4,))]:
        assert isinstance(value, np.ndarray) and value.shape == shape and np.isfinite(value).all()
    np.testing.assert_allclose(diag, [y.mean(), y.std(), end.mean(), end.std()])
    return y, state

verify(mod.solve_batch(study), len(study['test']['x']))
all_metrics = []
for name, i, j in splits:
    train = {k: p[k][i] for k in ('x', 'y', 'traj')}
    ytr, end = train['y'], train['traj'][:, -1]
    train['summary'] = np.array([ytr.mean(), ytr.std(), end.mean(), end.std()])
    out = mod.solve_batch({**study, 'train': train, 'test': {'x': x[j]}})
    y, state = verify(out, len(j)); err = abs(state - p['traj'][j]); m = (abs(y - p['y'][j]).mean(), err.mean(), err.mean(0).max())
    all_metrics.append(m)
    if name != 'random': print(name, np.round(m, 6))
worst = np.max(all_metrics, axis=0)
print('worst across 35 splits: scalar, trajectory, timestamp', np.round(worst, 6))
assert np.max(worst) <= 0.038
# Check that the shipped dead-zone operator, not just an exploratory fit, predicts omitted panel rows.
panel = study['panel_b']; u, v = panel['u'], panel['v']; offset, scale = mod._affine_reference(study['panel_a'])
errors = []; affine_errors = []
for j in range(len(u)):
    keep = np.arange(len(u)) != j
    reduced = {k: val[keep] for k, val in panel.items()}
    _, pred, _ = mod._nodes({**study, 'panel_b': reduced}, u[j:j+1] * scale + offset)
    design = np.column_stack((np.ones(len(u)), u))
    affine = design[j] @ np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
    errors.append(pred[0] - v[j]); affine_errors.append(affine - v[j])
rmse = np.sqrt(np.mean(np.square(errors))); affine_rmse = np.sqrt(np.mean(np.square(affine_errors)))
print('Panel B leave-one-row-out RMSE: dead zone', rmse, 'affine', affine_rmse)
assert rmse < 0.001
