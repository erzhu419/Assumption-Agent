import importlib.util
import pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', p+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
d = s['train']

def validate(o, n):
    y, t, diag = [o[k] for k in ('y_hat','state_pred','diagnostics_vec')]
    assert all(isinstance(a, np.ndarray) and np.isfinite(a).all() for a in (y,t,diag))
    assert (y.shape,t.shape,diag.shape) == ((n,), (n,8), (4,))
    np.testing.assert_array_equal(diag, [y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])
    return y, t

def check(ii, jj, **kw):
    a = {k:d[k][ii] for k in ('x','y','traj')}
    a['summary'] = np.array([a['y'].mean(),a['y'].std(),a['traj'][:,-1].mean(),a['traj'][:,-1].std()])
    o = m.solve_batch(dict(s, train=a, test={'x': d['x'][jj]}, **kw)) if not kw else m.solve_batch({**s, 'train':a, 'test':{'x':d['x'][jj]}, **kw})
    y, t = validate(o, len(jj))
    ey, et = abs(y-d['y'][jj]),abs(t-d['traj'][jj])
    return np.r_[ey.mean(),et.mean(),et.mean(0)]

ii, jj = np.arange(52),np.arange(52,72)
nominal = check(ii,jj)
print('specified 52/20: y, trajectory, and each horizon', nominal)
assert max(nominal) <= .038
rng = np.random.default_rng(128)
results = []
for _ in range(24):
    k = rng.permutation(72)
    results.append(check(k[:52],k[52:]))
print('worst of 24 random 52/20 slices', np.max(results,axis=0))
assert np.max(results) <= .038
z = m._calibrate(s['panel_a'],d['x'])
radius_order = np.argsort(np.linalg.norm(z,axis=1))
extreme = check(radius_order[:52], radius_order[52:])
print('largest-radius holdout',extreme)
assert max(extreme) <= .038
panels = []
for name in ('panel_a','panel_c'):
    for row in range(len(s[name]['u'])):
        panel = {k:np.delete(v,row,axis=0) for k,v in s[name].items()}
        panels.append(check(ii,jj,**{name:panel}))
print('worst reference/operator panel row jackknife',np.max(panels,axis=0))
assert np.max(panels) <= .038
validate(m.solve_batch(s), len(s['test']['x']))
