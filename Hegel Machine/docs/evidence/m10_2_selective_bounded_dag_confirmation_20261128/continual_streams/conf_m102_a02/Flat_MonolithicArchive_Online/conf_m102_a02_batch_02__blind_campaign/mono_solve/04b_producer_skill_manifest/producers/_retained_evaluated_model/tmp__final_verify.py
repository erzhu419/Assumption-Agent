import importlib.util
import pickle
import numpy as np

sp = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr = s['train']
ids = np.arange(len(tr['y']))
rng = np.random.default_rng(45)
sh,sc = m._calibrate(s['panel_a'])
z = (tr['x']-sh)/sc
l = m._aff(z)
h = np.sum(l * np.linalg.solve(l.T@l,l.T).T, axis=1)
splits = [('gate',ids[:52],ids[52:])]
for k in range(20):
    p = rng.permutation(ids)
    splits.append((f'random{k}',p[:52],p[52:]))
for k,u in enumerate([abs(z[:,2]-.42*z[:,0]),np.linalg.norm(z,axis=1),*z.T,*(-z.T),h]):
    p = np.argsort(u,kind='stable')
    splits.append((f'tail{k}',p[:-12],p[-12:]))

def check(o,n):
    a,b,d = (o[k] for k in ('y_hat','state_pred','diagnostics_vec'))
    for v,shape in zip((a,b,d),((n,),(n,len(s['eval_times'])),(4,))):
        assert isinstance(v,np.ndarray) and v.shape == shape and np.isfinite(v).all()
    np.testing.assert_allclose(d,[a.mean(),a.std(),b[:,-1].mean(),b[:,-1].std()],rtol=0,atol=1e-13)
    return a,b

check(m.solve_batch(s),len(s['test']['x']))
rows = []
for name,i,j in splits:
    ss = dict(s,train={k:(v[i] if k in ('x','y','traj') else v) for k,v in tr.items()},test={'x':tr['x'][j]})
    a,b = check(m.solve_batch(ss),len(j))
    e,r = abs(a-tr['y'][j]),abs(b-tr['traj'][j])
    rows.append((name,e.mean(),r.mean(),r.mean(axis=0).max()))
print('gate (slice, scalar MAE, state MAE, worst time MAE):',rows[0])
for k in range(1,4): print('worst',k,max(rows,key=lambda a:a[k]))
assert max(a[1] for a in rows) <= .038
print('validated',len(rows),'slices and full production output')
