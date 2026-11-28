import importlib.util
import pickle
import sys
import numpy as np
p = sys.argv[1] if len(sys.argv)>1 else '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
tr = s['train']; x = tr['x']; n = len(x); ix = np.arange(n)
w = m._calibration(s['panel_a']); a = m._inverse_calibration(x, w)
print('calibration inversion max error:', np.max(np.abs(np.einsum('nki,ki->nk', m._cal_design(a), w)-x)))
order = np.argsort(np.linalg.norm(a,axis=1))
slices = [('52/20', ix[52:], ix[:52]), ('56/16', ix[56:], ix[:56]), ('radius', order[-14:], order[:-14])]
for j in range(4):
    order = np.argsort(a[:,j]); v = np.r_[order[:6], order[-6:]]
    slices.append((f'edge{j}', v, np.setdiff1d(ix,v)))
for j, v in enumerate(np.array_split(np.random.default_rng(7).permutation(n),4)):
    slices.append((f'random{j}',v,np.setdiff1d(ix,v)))
for name,v,f in slices:
    out = m.solve_batch(dict(s, train={k:tr[k][f] for k in ('x','y','traj')}, test={'x':x[v]}))
    y = np.mean(np.abs(out['y_hat']-tr['y'][v])); z = np.mean(np.abs(out['state_pred']-tr['traj'][v]),axis=0)
    print(f'{name:8s} scalar {y:.6f} trajectory {z.mean():.6f} worst time {z.max():.6f}')
    assert y <= .038 and z.max() <= .038
