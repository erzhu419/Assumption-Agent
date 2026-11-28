import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
d = s['train']; x = d['x']; y = d['y']; z = d['traj']; n = len(x)
rng = np.random.default_rng(2951)
sets = [np.arange(56,72)] + [rng.permutation(n)[:16] for _ in range(12)] + [np.argsort(abs(x[:, j]))[-16:] for j in range(4)]
for j in range(4):
    sets.append(np.argsort(np.linalg.norm(x - x[np.argmax(abs(x[:,j]))], axis=1))[:16])
errs = []
for te in sets:
    tr = np.setdiff1d(np.arange(n), te)
    pred = m.solve_batch(dict(s, train={k: d[k][tr] for k in ('x','y','traj')}, test={'x':x[te]}))
    e = abs(pred['state_pred'] - z[te]); errs.append([np.mean(abs(pred['y_hat']-y[te])), e.mean(), e[:,-1].mean()])
e = np.asarray(errs)
for name,a in [('56/16', e[:1]), ('random',e[1:13]), ('boundaries', e[13:17]), ('corner neighborhoods',e[17:])]:
    print(name,'mean',a.mean(0),'worst',a.max(0))
p = m.solve_batch(s); q = p['y_hat']; zt = p['state_pred']; f = zt[:,-1]
assert q.shape == (36,) and zt.shape == (36,8)
assert all(np.isfinite(a).all() for a in p.values())
np.testing.assert_allclose(p['diagnostics_vec'],[q.mean(),q.std(),f.mean(),f.std()])
