import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; z=m._calibration(s['panel_a'],tr['x']); q=m._quadratic(z); a=m._affine(z); B=m._fit(m._affine(s['panel_b']['u']),s['panel_b']['v']); bz=a@B; n=len(y); whole=np.arange(n)
splits=[('52/20',whole[:52],whole[52:]),('56/16',whole[:56],whole[56:])]+[(f'fold{i}',np.setdiff1d(whole,j),j) for i,j in enumerate(np.array_split(np.random.default_rng(122).permutation(n),5))]+[('outer-range',np.argsort(np.linalg.norm(z,axis=1))[:-14],np.argsort(np.linalg.norm(z,axis=1))[-14:])]
for label,i,j in splits:
 print(label,'traj MAE',np.abs(q[j]@m._fit(q[i],T[i])-T[j]).mean(),end=' ')
 for k in ('tanh','rat','sin'):
  p,w,f=m._bounded_fit(z[i],y[i],k,starts=12); error=np.abs(np.c_[q,f(a@p)]@w-y); print(k,np.round([error[i].mean(),error[j].mean(),error[j].max()],5),end=' ')
  if label=='52/20':
   wb=m._fit(np.c_[m._quadratic(bz[i]),f(a[i]@p)],y[i]);print('index',p.round(3),'z readout',w.round(3),'B readout',wb.round(3),end=' ')
 print()
