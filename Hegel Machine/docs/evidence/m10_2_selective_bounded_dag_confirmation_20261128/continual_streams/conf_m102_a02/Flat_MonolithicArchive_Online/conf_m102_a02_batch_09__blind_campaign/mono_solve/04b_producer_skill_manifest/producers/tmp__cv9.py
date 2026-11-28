import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']);rng=np.random.default_rng(910)
cal=m._calibrate(s['panel_a']); print('calibration',cal); x=m._inverse(tr['x'],cal); q=m._inverse(s['test']['x'],cal); print('x/query extrema',x.min(0),x.max(0),q.min(0),q.max(0));
y=tr['y']; z=tr['traj']; f=m._features(x,m._rational_panel(s['panel_c'])); print('scalar coef',m._ls(f,y));print('state sv',np.linalg.svd(m._ls(m._aff(x),z),compute_uv=False)); print('time',s['eval_times'],'telemetry',s['telemetry_vec']);
maes=[]
for split in [np.r_[np.arange(52),np.arange(52,n)]]+[rng.permutation(n) for i in range(24)]+[np.argsort(np.linalg.norm(x,axis=1))]+[np.argsort(sign*x[:,i]) for i in range(4) for sign in [-1,1]]:
 a,b=split[:52],split[52:]; d={**s,'train':{**tr,**{k:tr[k][a] for k in ('x','y','traj')}},'test':{'x':tr['x'][b]}}; p=m.solve_batch(d); ey=abs(p['y_hat']-y[b]); es=abs(p['state_pred']-z[b]); maes.append([ey.mean(),es.mean(),es[:,-1].mean(),ey.max(),es.max()]);
print('fixed (y/traj/end/max-y/max-traj)',maes[0]); print('worst random',np.max(maes[1:25],axis=0));print('worst extrapolation',np.max(maes[25:],axis=0));
