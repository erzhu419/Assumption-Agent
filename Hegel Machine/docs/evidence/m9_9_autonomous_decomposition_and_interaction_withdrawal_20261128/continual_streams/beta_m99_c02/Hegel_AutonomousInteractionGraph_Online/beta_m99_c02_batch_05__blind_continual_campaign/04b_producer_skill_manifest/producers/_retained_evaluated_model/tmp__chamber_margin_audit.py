import runpy
import sys
import numpy as np
from scipy.sparse import diags
sys.path[:0] = ['/holdout_task', '/tmp']
from chamber_probe import data, response
from downstream_solver import _axes, _responses

s, W, Y = data()
ax, ay = _axes(s)
S, _ = _responses(s)
E = S @ W - Y
for A in (ax, ay):
    np.testing.assert_allclose(A @ np.ones(A.shape[0]), 0, atol=1e-11, rtol=0)
np.testing.assert_allclose(response(s, ax, ay), S, atol=1e-11, rtol=0)

# Profile small Robin leakage independently at all four endpoints, including
# leakage superposed on the x seam. Fit only on the retained sensor blocks.
columns = []
step = 1e-4
for a, A in enumerate((ax, ay)):
    for end in (0, A.shape[0]-1):
        d = np.zeros(A.shape[0]); d[end] = step
        D = diags(d, format='csc')
        plus = (ax+D, ay) if a == 0 else (ax, ay+D)
        minus = (ax-D, ay) if a == 0 else (ax, ay-D)
        columns.append(((response(s, *plus) - response(s, *minus)) @ W)/(2*step))
G = np.stack(columns, axis=-1)
Z = s['Z_nuisance']
n = len(E)
width = len(s['sensor_ij'])
all_rows = np.arange(n)
for name, group in [('bank', all_rows % 4), ('shift', all_rows // width)]:
    base, fit, max_rate = [], [], 0.
    for k in np.unique(group):
        keep = group != k
        z = Z[keep]
        P = np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
        y = E[keep]
        g = G[keep]
        gy = P @ y
        gg = np.einsum('ij,jpk->ipk', P, g)
        x = g - np.einsum('ij,jpk->ipk', z, gg)
        rate = np.linalg.lstsq(x.reshape(-1,4), -(y-z@gy).ravel(), rcond=None)[0]
        r = E[~keep] - Z[~keep] @ gy
        J = G[~keep] - np.einsum('ij,jpk->ipk', Z[~keep], gg)
        base.append(np.sqrt(np.mean(r**2)))
        fit.append(np.sqrt(np.mean((r+J@rate)**2)))
        max_rate = max(max_rate, np.max(np.abs(rate)))
    print(f'{name}: profiled held-out RMSE fixed={np.array(base)}, extra leakage={np.array(fit)}; max abs fitted leakage={max_rate:.6g}')

sys.argv = ['validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path('/holdout_task/validate_public_schema.py', run_name='__main__')
