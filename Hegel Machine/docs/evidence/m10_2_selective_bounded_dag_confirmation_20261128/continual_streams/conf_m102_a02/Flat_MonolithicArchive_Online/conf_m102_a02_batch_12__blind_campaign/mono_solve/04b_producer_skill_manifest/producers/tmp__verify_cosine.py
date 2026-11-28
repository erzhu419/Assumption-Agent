import importlib.util, pickle, numpy as np
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']
a,b=m._calibrate(s['panel_a']); pp=s['panel_b']; op=m._operator(pp)
print('panel B in-sample RMSE', np.sqrt(np.mean((op(pp['u'])-pp['v'])**2)))
loo=[]
for i in range(len(pp['u'])):
    mask=np.arange(len(pp['u']))!=i
    pred=m._operator({k:v[mask] for k,v in pp.items()})(pp['u'][i:i+1]); loo.append(np.mean(abs(pred-pp['v'][i])))
print('panel B leave-one-out mean/max MAE', np.mean(loo),np.max(loo))
def run(label, train, test):
    d=dict(s, train={**{k:tr[k][train] for k in ('x','y','traj')},'summary':np.zeros(4)},test={'x':x[test]})
    r=m.solve_batch(d); e=abs(r['y_hat']-tr['y'][test]); f=abs(r['state_pred']-tr['traj'][test]); print(label,len(train),len(test),'y/traj MAE',e.mean(),f.mean(),'worst row y/traj',e.max(),f.mean(1).max(),'worst time',f.mean(0).max()); assert max(e.mean(), f.mean(), e.max(), f.mean(1).max(), f.mean(0).max())<.038
run('prescribed', np.arange(52),np.arange(52,72))
for seed in range(10):
    order=np.random.default_rng(seed).permutation(len(x)); run('random'+str(seed),order[:52],order[52:])
for j in range(x.shape[1]):
    index=np.argsort(abs(x[:,j]-a[j])); run('axis-tail'+str(j),index[:52],index[52:])
r=m.solve_batch(s); y=r['y_hat']; z=r['state_pred']; assert y.shape==(36,) and z.shape==(36,8) and all(np.all(np.isfinite(v)) for v in r.values()); np.testing.assert_array_equal(r['diagnostics_vec'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); print('test contract OK')
