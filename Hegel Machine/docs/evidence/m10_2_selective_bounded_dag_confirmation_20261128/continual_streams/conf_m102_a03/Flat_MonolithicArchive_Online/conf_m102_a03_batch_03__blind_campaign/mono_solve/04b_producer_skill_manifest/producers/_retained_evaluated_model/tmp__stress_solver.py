import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; p,c=m._operator(s['panel_b']); o,g=m._calibration(s['panel_a']); z=(tr['x']-o)/g; w=m._affine(m._warp(z,p))@c
x=np.arange(len(z)); T=tr['traj']; y=tr['y']; ts=s['eval_times']
print('calibration',o,g,'operator',p,'scalar coef',m._fit(m._readout(w),y)); print('traj spatial residual rms',np.sqrt(np.mean((T-w@m._fit(w,T))**2,axis=0)))

def check(ho, s=s):
 ix=np.setdiff1d(x,ho); trial=dict(s,train={k:tr[k][ix] for k in ('x','y','traj')},test={'x':tr['x'][ho]}); sol=m.solve_batch(trial)
 ey=np.abs(sol['y_hat']-y[ho]); et=np.abs(sol['state_pred']-T[ho]); last=sol['state_pred'][:,-1]; np.testing.assert_array_equal(sol['diagnostics_vec'],[sol['y_hat'].mean(),sol['y_hat'].std(),last.mean(),last.std()]); return np.r_[ey.mean(),et.mean(),et.mean(axis=0)]
print('52/20 y/traj/per-time',check(x[52:]))
R=np.array([check(np.random.default_rng(i).permutation(x)[52:]) for i in range(64)]); print('64 splits worst',R.max(axis=0))
S=[]
for j in range(4):
 idx=np.argsort(z[:,j]); S.extend([check(idx[:20]),check(idx[-20:])])
for v in [np.linalg.norm(z,axis=1),np.count_nonzero(abs(z)>p[0],axis=1),np.min(abs(abs(z)-p[0]),axis=1)]:
 idx=np.argsort(v); S.extend([check(idx[:20]),check(idx[-20:])])
print('coordinate tails, activation and threshold slices worst',np.max(S,axis=0))
for panel in ['panel_a','panel_b']:
 vals=[]
 for i in range(len(s[panel]['u'])):
  small=dict(s,**{panel:{k:np.delete(v,i,axis=0) for k,v in s[panel].items()}}); vals.append(check(x[52:],small))
 print(panel,'jackknife worst',np.max(vals,axis=0))
