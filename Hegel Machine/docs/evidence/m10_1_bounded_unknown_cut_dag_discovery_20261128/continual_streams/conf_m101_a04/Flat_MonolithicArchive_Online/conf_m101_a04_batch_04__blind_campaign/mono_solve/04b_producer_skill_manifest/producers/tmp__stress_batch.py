import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=(tr[k] for k in ('x','y','traj')); t=s['eval_times']; f,g,h=m._features(s,x)
i=np.arange(len(x)); rng=np.random.default_rng(835)
splits=[('public',i[52:])]+[(f'random{j}',rng.permutation(i)[:16]) for j in range(12)]
x_cal=m._calibrate(s['panel_a'],x)
splits.append(('large_norm',np.argsort(np.linalg.norm(x_cal,axis=1))[-16:]))
for j in range(4):
 order=np.argsort(x_cal[:,j]); splits.extend([(f'x{j}_low',order[:16]),(f'x{j}_high',order[-16:])])
metrics=[]
for name,b in splits:
 a=np.setdiff1d(i,b); local={**s,'train':{'x':x[a],'y':y[a],'traj':z[a]},'test':{'x':x[b]}}
 out=m.solve_batch(local); ey=np.abs(out['y_hat']-y[b]); ez=np.abs(out['state_pred']-z[b]); vals=np.r_[ey.mean(),ez.mean(),ez.mean(0).max()]; metrics.append(vals); print(name,*np.round(vals,6))
print('worst',np.max(metrics,axis=0)); print('scalar weights',np.round(m._fit(f,y),5)); rates,w=m._fit_trajectory(g,h,t,z); print('trajectory rates',rates,'weights',np.round(w,5))
