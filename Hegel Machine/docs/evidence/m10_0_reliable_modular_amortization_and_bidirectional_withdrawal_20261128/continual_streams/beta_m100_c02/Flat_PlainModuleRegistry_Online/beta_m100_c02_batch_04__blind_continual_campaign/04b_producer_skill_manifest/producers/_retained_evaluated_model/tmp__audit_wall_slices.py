import sys
sys.path[:0] = ['/holdout_task', '/tmp']
import numpy as np
from fit_walls import s, ps, target, predict
from batch_solver import solve_batch
from downstream_solver import chamber_response
import pickle

S, fields = chamber_response(s)
W = np.column_stack([p['w_pulse'] for p in ps])
y = S @ W
err = y - target
k = s['kappa_x']; v = s['vel_x']; h = s['Lx']/(len(k)-1)
end = k[[0, -1]]
face = (end + k[[1, -2]]) / 2
near = face + h*np.maximum([-v[0], v[-1]], 0)
seam = end.mean() + h*np.maximum([v[0], -v[-1]], 0)
full = np.r_[near/end, 0., 0., seam/end, 1., 1., 0., 0., 0., 0.]
np.testing.assert_allclose(y, predict(full), atol=2e-13, rtol=0)
variants = {'no seam': (slice(4,6), [0.,0.]), 'diffusive seam only': (slice(4,6), end.mean()/end), 'endpoint kappa in place of face': (slice(0,2), [1.,1.]), 'open y': (slice(8,10), [1.,1.])}
Z = s['Z_nuisance']
P = Z @ np.linalg.pinv(Z)
R = np.eye(len(Z)) - P
rms = lambda a: np.sqrt(np.mean(a*a,axis=0))
blocks = np.array_split(np.arange(len(Z)), len(s['mu_shifts']))
bank = np.asarray(s['row_bank_id'])
print('Periodic x / closed y pulse RMSE:', rms(err))
print('Mean pulse residual by bank:', {str(b): err[bank==b].mean(axis=0).tolist() for b in np.unique(bank)})
print('Nuisance-orthogonal pulse RMSE:', rms(R @ err))
worst = max(float(rms(err[idx[bank[idx]==b]]).max()) for idx in blocks for b in np.unique(bank[idx]))
print('Worst pulse × shift × bank RMSE:', worst)
for name,(cols,val) in variants.items():
    q = full.copy(); q[cols] = val
    alt = predict(q) - target
    print(name, 'raw pulse RMSE', rms(alt), 'nuisance-orthogonal pulse RMSE', rms(R @ alt))
    assert rms(R @ alt).min() > 10 * rms(R @ err).max()
assert worst < 5e-6
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    out = solve_batch(pickle.load(f)[0]['study'])
c = out['kkt_certificate']
print('Isolated batch entry point: field shape', out['terminal_field'].shape, 'duality gap', c['primal_objective'] - c['dual_lower_bound'])
assert out['terminal_field'].shape == fields.shape[1:]
assert abs(c['primal_objective'] - c['dual_lower_bound']) < 1e-9
