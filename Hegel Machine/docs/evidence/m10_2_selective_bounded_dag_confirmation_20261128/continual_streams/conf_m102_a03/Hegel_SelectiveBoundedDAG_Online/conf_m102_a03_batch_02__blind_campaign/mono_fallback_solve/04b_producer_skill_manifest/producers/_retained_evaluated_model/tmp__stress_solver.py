import pickle, runpy, numpy as np
solve=runpy.run_path('/holdout_task/batch_solver.py')['solve_batch']
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); rng=np.random.default_rng(512)
def check(i,j,label):
    v=solve(dict(s,train={**tr,**{k:tr[k][i] for k in ('x','y','traj')}},test={'x':tr['x'][j]}))
    dy=abs(v['y_hat']-tr['y'][j]); dt=abs(v['state_pred']-tr['traj'][j]); final=v['state_pred'][:,-1]
    np.testing.assert_allclose(v['diagnostics_vec'],[v['y_hat'].mean(),v['y_hat'].std(),final.mean(),final.std()]); assert np.isfinite(dy).all() and np.isfinite(dt).all()
    print(label, 'y',round(dy.mean(),6),'trajectory',round(dt.mean(),6),'worst horizon',round(dt.mean(0).max(),6),flush=True)
    return dy.mean(),dt.mean(),dt.mean(0).max()
metrics=[check(np.arange(52),np.arange(52,n),'public')]
for k in range(10):
    p=rng.permutation(n); metrics.append(check(p[:52],p[52:],f'random{k}'))
for name,value in [('radius',np.linalg.norm(tr['x']-tr['x'].mean(0),axis=1)),('y-index',np.abs(tr['x']@np.linalg.lstsq(tr['x'],tr['y'],rcond=None)[0])),('state',abs(tr['traj'][:,-1]))]:
    p=np.argsort(value); metrics.append(check(p[:52],p[52:],name))
print('worst',np.max(metrics,axis=0)); out=solve(s); print('test shapes', {k:v.shape for k,v in out.items()}); assert all(np.isfinite(v).all() for v in out.values())
