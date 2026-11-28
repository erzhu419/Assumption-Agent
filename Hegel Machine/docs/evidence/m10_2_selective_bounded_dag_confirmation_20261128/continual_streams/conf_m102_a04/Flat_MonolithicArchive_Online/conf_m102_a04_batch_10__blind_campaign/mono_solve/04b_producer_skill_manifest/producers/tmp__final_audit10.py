import importlib.util, pickle, numpy as np

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
base=load('base','/tmp/batch10_full_bilinear.py')
m=load('final','/holdout_task/batch_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; n=len(tr['x']); z=m._calibrate(s['panel_a'],tr['x'])
groups={'public':np.arange(52,n),'high_radius':np.argsort(np.linalg.norm(z,axis=1))[-20:]}
for i in range(4):
    groups[f'axis_{i}_low']=np.argsort(z[:,i])[:20]
    groups[f'axis_{i}_high']=np.argsort(z[:,i])[-20:]
for i in range(8): groups[f'random_{i}']=np.random.default_rng(i).permutation(n)[:20]
allm=[]
for name,hold in groups.items():
    fit=np.setdiff1d(np.arange(n),hold)
    q=dict(s,train={**tr,**{k:tr[k][fit] for k in ('x','y','traj')}},test={'x':tr['x'][hold]})
    rows=[]
    for solver in [m,base]:
        o=solver.solve_batch(q); y=o['y_hat']; st=o['state_pred']; e=np.abs(st-tr['traj'][hold]); v=st[:,-1]
        assert y.shape==(len(hold),) and st.shape==(len(hold),len(s['eval_times']))
        assert np.isfinite(y).all() and np.isfinite(st).all()
        np.testing.assert_allclose(o['diagnostics_vec'],[y.mean(),y.std(),v.mean(),v.std()],rtol=0,atol=1e-12)
        rows.append([np.mean(np.abs(y-tr['y'][hold])),e.mean(),e.mean(0).max(),e[:,-1].mean()])
    allm.append(rows); print(name, np.round(rows,6).tolist())
print('Worst MAEs [y, trajectory, horizon, terminal]; compact then full:',np.max(allm,axis=0))
assert np.max(np.array(allm)[:,0]) < .038
