import importlib.util,pickle,numpy as np
spc=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spc);spc.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train']; n=len(t['x']); J=np.arange(n); z=m._decode(t['x'],m._calibration(s['panel_a'])); p,b=m._operator(s['panel_b']); w=m._affine(m._warp(z,p))@b
S=[(J[:52],J[52:]),(J[20:],J[:20])]; rng=np.random.default_rng(42)
for _ in range(8):
 r=rng.permutation(n);S.append((r[:52],r[52:]))
for a in np.r_[z.T,-z.T]:
 r=np.argsort(a); S.append((r[:-12],r[-12:]))
E=[]
for i,j in S:
 ss=dict(s,train={k:v[i] if k!='summary' else np.array([t['y'][i].mean(),t['y'][i].std(),t['traj'][i,-1].mean(),t['traj'][i,-1].std()]) for k,v in t.items()},test={'x':t['x'][j]}); r=m.solve_batch(ss); E.append([np.abs(r['y_hat']-t['y'][j]).mean(),np.abs(r['state_pred']-t['traj'][j]).mean()]); assert np.allclose(r['diagnostics_vec'],[r['y_hat'].mean(),r['y_hat'].std(),r['state_pred'][:,-1].mean(),r['state_pred'][:,-1].std()])
print('CV primary, worst nominal, worst tail:',np.array(E[0]),np.max(E[:10],0),np.max(E[10:],0)); print(np.array(E).round(5)); assert E[0][0]<=.038
