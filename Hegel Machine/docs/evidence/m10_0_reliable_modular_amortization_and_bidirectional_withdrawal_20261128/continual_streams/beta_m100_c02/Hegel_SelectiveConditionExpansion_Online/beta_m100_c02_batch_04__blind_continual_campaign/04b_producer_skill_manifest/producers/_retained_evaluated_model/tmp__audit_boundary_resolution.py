import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import axis_operator, chamber_response

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
cal = calibrate_upstream(dict(st, reference_tracer_pulses=item['campaign_diagnostics']))
pulses = cal['reference_tracer_pulses']
w = np.column_stack([p['w_pulse'] for p in pulses])
y = np.column_stack([p['y_pulse_raw'] for p in pulses])
z = st['Z_nuisance']
nx, ny = st['nx'], st['ny']
hx, hy = (st['L'+a] / (st['n'+a] - 1) for a in 'xy')
k, v = st['kappa_x'], st['vel_x']
ax = axis_operator(k, v, hx, True).toarray()
ay = axis_operator(st['kappa_y'], st['vel_y'], hy, False).toarray()
row_error = max(np.max(np.abs(ax.sum(axis=1))), np.max(np.abs(ay.sum(axis=1))))
assert row_error < 1e-12
s, _ = chamber_response(cal)
full = s @ w
rhs = np.stack([m.ravel(order='F') for m in st['source_modes']], axis=1) @ w
ij = st['sensor_ij']
idx = ij[:, 0] + nx * ij[:, 1]
fixed = np.kron(ay, np.eye(nx)) + np.diag(st['V_reaction_2d'].ravel(order='F'))
shifts = st['mu_shifts']
eye = np.eye(nx * ny)
def predict(a):
    l = fixed + np.kron(np.eye(ny), a)
    return np.concatenate([np.linalg.solve(l + mu * eye, rhs)[idx] for mu in shifts])
np.testing.assert_allclose(predict(ax), full, rtol=0, atol=1e-12)
seam = (k[0] + k[-1]) / (2 * hx**2)
variants = {'exact': full}
for name in ('no_cross_seam', 'no_seam_upwind', 'no_endpoint_face_average'):
    a = ax.copy()
    for i, far, near in ((0, nx-1, 1), (nx-1, 0, nx-2)):
        if name == 'no_cross_seam':
            a[i, far] = 0
        elif name == 'no_seam_upwind':
            a[i, far] = -seam
        else:
            a[i, near] += (k[near] - k[i]) / (2 * hx**2)
        a[i, i] = 0
        a[i, i] = -a[i].sum()
    variants[name] = predict(a)
print('constant-field transport residual Linf:', row_error)
# Hold out each sensor rosette. Fit only the nuisance coefficients on the
# remaining rows; the boundary and calibrated forward response stay fixed.
for name, pred in variants.items():
    raw = pred - y
    corrected = np.empty_like(raw)
    for start in range(0, len(y), 4):
        test = np.arange(start, start+4)
        train = np.ones(len(y), dtype=bool)
        train[test] = False
        coef = np.linalg.lstsq(z[train], raw[train], rcond=None)[0]
        corrected[test] = raw[test] - z[test] @ coef
    def worst(error):
        return (np.linalg.norm(error.reshape(-1, 4, w.shape[1]), axis=1) / np.linalg.norm(y.reshape(-1, 4, w.shape[1]), axis=1)).max()
    print(name, 'worst rosette relative L2 raw:', worst(raw), 'after held-out nuisance removal:', worst(corrected))
    if name == 'exact':
        assert worst(raw) < 3e-4 and worst(corrected) < 3e-4
    else:
        assert worst(corrected) > 1e-3
