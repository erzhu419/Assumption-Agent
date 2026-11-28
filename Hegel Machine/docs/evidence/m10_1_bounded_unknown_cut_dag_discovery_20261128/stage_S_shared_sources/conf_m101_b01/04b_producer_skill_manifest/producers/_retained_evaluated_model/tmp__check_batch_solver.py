import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
tr=s['train']; x=m._coordinates(s,tr['x']); n=len(x); rng=np.random.default_rng(413)
splits={'ordered':np.arange(n),'radius':np.argsort(np.linalg.norm(x,axis=1))}
for k in range(4):
    for sign in (1,-1): splits[f'axis{k}/{sign}']=np.argsort(sign*x[:,k])
for k in range(13): splits[f'random{k}']=rng.permutation(n)
def check(args):
    out=m.solve_batch(args); y,z,d=(out[k] for k in ('y_hat','state_pred','diagnostics_vec'))
    assert (y.shape,z.shape,d.shape)==((len(args['test']['x']),),(len(y),len(args['eval_times'])),(4,))
    assert all(np.isfinite(a).all() for a in (y,z,d))
    np.testing.assert_array_equal(d,[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); return y,z
rows={}
for name,idx in splits.items():
    i,j=idx[:52],idx[52:]; sub=dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':tr['x'][j]})
    y,z=check(sub); e=np.abs(z-tr['traj'][j]).mean(0)
    rows[name]=(np.abs(y-tr['y'][j]).mean(),e.mean(),e.max())
    if not name.startswith('random'): print(name,np.round(rows[name],6))
    if name=='ordered':
        print('horizons',np.round(e,6)); assert rows[name][0]<=0.038
print('random worst',np.max([v for k,v in rows.items() if k.startswith('random')],axis=0))
print('all-slice worst',[(max(rows,key=lambda k:rows[k][i]),max(v[i] for v in rows.values())) for i in range(3)])
check(s)
