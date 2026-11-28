import importlib.util, pickle, numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];t=s['eval_times'];n=len(tr['y']); ix=np.arange(n)
splits=[ix[52:],ix[:20],ix[20:40],ix[40:60]]+[np.random.default_rng(i).choice(n,20,False) for i in range(12)]
metrics=[]
for jj,test in enumerate(splits):
 fit=np.setdiff1d(ix,test); w=dict(s,train={k:v[fit] for k,v in tr.items() if k!='summary'},test={'x':tr['x'][test]});o=m.solve_batch(w)
 ey=np.abs(o['y_hat']-tr['y'][test]); et=np.abs(o['state_pred']-tr['traj'][test]); radius=np.linalg.norm(m._calibrate(s['panel_a'],tr['x'][test]),axis=1); hi=np.argsort(radius)[-5:]; vals=[ey.mean(),et.mean(),et[:,-1].mean(),et.mean(0).max(),ey[hi].mean(),et[hi].mean(),ey.max(),et.max()]; metrics.append(vals)
 if jj<4: print(jj,np.round(vals,6))
print('worst across splits:',np.round(np.max(metrics,axis=0),6)); assert max(metrics[0][:3])<=.038
z=m._calibrate(s['panel_a'],np.r_[tr['x'],s['test']['x']]);p,q=m._branches(s['panel_c'],z);F=m._design(np.c_[p,q,(p*q).sum(1),p*np.roll(q,-1,axis=1)]);W=m._ls(F[:52],tr['traj'][:52]); H=m._relaxation(t,tr['traj'][:52]);print('W singular values',np.linalg.svd(W,compute_uv=False)); print('coefficient residual after projection',np.round(W-(H@m._ls(H,W.T)).T,4));print('mode coefficients',np.round(m._ls(H,W.T),4));
