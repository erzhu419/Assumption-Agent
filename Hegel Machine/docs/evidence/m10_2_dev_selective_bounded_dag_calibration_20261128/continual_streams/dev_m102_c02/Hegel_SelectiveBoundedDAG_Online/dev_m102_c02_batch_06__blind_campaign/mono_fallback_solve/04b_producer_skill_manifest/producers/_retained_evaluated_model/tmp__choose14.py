import importlib.util,pickle,numpy as np
path='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('bs',path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(1865); R=[]; W=np.array([0.,.5,1.]);
# Use independent randomized stress splits to select whether the new readout transfers.
for p in [rng.permutation(72) for _ in range(6)]+[np.arange(72)]:
 i,j=p[:52],p[52:]; t=s['train']; b=dict(s,train={**t,**{k:t[k][i] for k in ('x','y','traj')}},test={'x':t['x'][j]});x=m._calibrate(b['panel_a'],np.r_[t['x'][i],t['x'][j]]); old,state=m._readout(b,x); new=m._fit_readout_operators(b,x); e=np.mean(abs((1-W[:,None])*old+W[:,None]*new-t['y'][j]),axis=1); R.append(e);print('y MAE by blend',e,'state MAE',np.mean(abs(state-t['traj'][j])),flush=True)
R=np.array(R); v=R[:-1]; score=.8*v.mean(0)+.2*v.max(0); k=np.argmin(score); w=W[k]; print('selected weight',w,'random mean',v.mean(0),'worst',v.max(0),'fixed',R[-1]);
if w:
 with open(path,'a') as f:
  f.write('\n\ndef solve_batch(study: dict) -> dict:\n    tr=study[\'train\']; x=_calibrate(study[\'panel_a\'],np.r_[tr[\'x\'],study[\'test\'][\'x\']]); y,p=_readout(study,x)\n    y=(1-'+str(w)+')*y+'+str(w)+'*_fit_readout_operators(study,x)\n    return {\'y_hat\':y,\'state_pred\':p,\'diagnostics_vec\':np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])}\n')
else:
 with open(path,'w') as f:f.write(open('/tmp/batch_v02.py').read())
