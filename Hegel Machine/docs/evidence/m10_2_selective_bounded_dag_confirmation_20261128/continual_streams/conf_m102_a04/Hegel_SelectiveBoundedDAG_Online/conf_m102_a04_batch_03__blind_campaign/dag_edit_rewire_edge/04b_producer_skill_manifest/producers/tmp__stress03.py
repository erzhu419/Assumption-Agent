exec(open('/tmp/inspect03.py').read().split("print('times'")[0])
import runpy
m=runpy.run_path('/holdout_task/node_03.py'); off,sc=a._align(s['panel_a']); worst=np.zeros(2)
for seed in [-2,-1,*range(8)]:
 idx=np.arange(n) if seed<0 else np.random.default_rng(seed).permutation(n); k=n if seed==-2 else 52; i,j=idx[:k],idx[k:]; x=s['test']['x'] if seed==-2 else tr['x'][j]; ss={**s,'train':{k:tr[k][i] for k in ('x','y','traj')},'test':{'x':x}}; W=a.run_node({'__input__':ss}); T=b.run_node({'__input__':ss}); C=c.run_node({'node_00':W,'node_01':T}); z,loss=m['_decode'](W['features'],len(i)); real=(np.vstack((tr['x'][i],x))-off)/sc; print('latent',seed,loss,np.max(abs(z-real)),flush=True); pred=m['run_node']({'node_00':W,'node_01':T,'node_02':C}); assert all(np.all(np.isfinite(v)) for v in pred.values()); state=pred['state_pred']; y=pred['y_hat']; assert np.allclose(pred['diagnostics_vec'],[y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()]);
 if seed!=-2:
  err=[np.mean(abs(y-tr['y'][j])),np.mean(abs(state-tr['traj'][j]))]; worst=np.maximum(err,worst); print('mae',seed,err,flush=True)
print('worst',worst)
