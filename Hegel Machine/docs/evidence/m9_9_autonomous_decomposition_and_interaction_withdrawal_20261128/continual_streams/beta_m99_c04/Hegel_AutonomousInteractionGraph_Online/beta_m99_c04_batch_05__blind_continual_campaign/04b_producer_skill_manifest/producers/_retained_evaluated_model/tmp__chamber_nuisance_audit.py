import contextlib
import io
import runpy
import sys

import numpy as np

sys.path.insert(0, '/holdout_task')
from downstream_solver import _design

with contextlib.redirect_stdout(io.StringIO()):
    p = runpy.run_path('/tmp/wall_probe.py')
s = p['s']
errors = {(a, b): p['residual'](params) for _, a, b, params in p['rank']}
selected = ('periodic', 'sealed')
Z = s['Z_nuisance'].reshape(*errors[selected].shape[:2], -1)
S, _ = _design(s)
shipped = (S @ p['W'] - p['Y']).reshape(errors[selected].shape)
np.testing.assert_allclose(shipped, errors[selected], rtol=0, atol=1e-12)

def nuisance_free(r, z):
    a = r.reshape(-1, r.shape[-1])
    x = z.reshape(len(a), -1)
    return a - x @ np.linalg.lstsq(x, a, rcond=None)[0]

def rmse(r):
    return np.sqrt(np.mean(r * r))

worst_held = 0.0
min_train_margin = np.inf
min_held_margin = np.inf
for shift in range(shipped.shape[0]):
    tr_s = np.arange(shipped.shape[0]) != shift
    for pulse in range(shipped.shape[2]):
        tr_p = np.arange(shipped.shape[2]) != pulse
        train = {k: rmse(nuisance_free(r[tr_s][:, :, tr_p], Z[tr_s])) for k, r in errors.items()}
        assert min(train, key=train.get) == selected, (shift, pulse, train)
        train_margin = min(v for k, v in train.items() if k != selected) - train[selected]
        held = {k: rmse(nuisance_free(r[shift:shift+1, :, pulse:pulse+1], Z[shift:shift+1])) for k, r in errors.items()}
        held_margin = min(v for k, v in held.items() if k != selected) - held[selected]
        min_train_margin = min(min_train_margin, train_margin)
        min_held_margin = min(min_held_margin, held_margin)
        worst_held = max(worst_held, held[selected])
assert worst_held < 1e-5, worst_held
print(f'Crossed shift/pulse holdout after unregularized nuisance removal: all {shipped.shape[0]*shipped.shape[2]} training folds select periodic-x/sealed-y; worst held RMSE {worst_held:.6g}; minimum RMSE advantage on training folds {min_train_margin:.6g}, on held slices {min_held_margin:.6g}')
