import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; rng=np.random.default_rng(905)
q=mod._coordinates(s,x); d=np.linalg.norm(q[:,:4]-np.median(q[:,:4],0),axis=1)
splits={'52/20':np.arange(52,72),'reverse20':np.arange(20),'outer radial':np.argsort(d)[-20:]}
for i in range(8):splits[f'random{i}']=rng.permutation(72)[52:]
worst=np.zeros(2)
for name,te in splits.items():
 ti=np.setdiff1d(np.arange(72),te); o=mod.solve_batch({**s,'train':{**tr,'x':x[ti],'y':y[ti],'traj':z[ti]},'test':{'x':x[te]}})
 a,b=abs(o['y_hat']-y[te]),abs(o['state_pred']-z[te]); print(name,'mae',a.mean(),b.mean(),'state by time',b.mean(0).round(5),'max sample',a.max(),b.mean(1).max()); worst=np.maximum(worst,[a.mean(),b.mean()]);
 if name=='52/20':assert max(a.mean(),b.mean())<=.038
print('worst split:',worst)
o=mod.solve_batch(s); assert all(np.all(np.isfinite(v)) for v in o.values()); yy=o['y_hat']; zz=o['state_pred']; assert yy.shape==(36,) and zz.shape==(36,8); np.testing.assert_array_equal(o['diagnostics_vec'],[yy.mean(),yy.std(),zz[:,-1].mean(),zz[:,-1].std()]); print('clean process: full prediction finite, exact diagnostics')
