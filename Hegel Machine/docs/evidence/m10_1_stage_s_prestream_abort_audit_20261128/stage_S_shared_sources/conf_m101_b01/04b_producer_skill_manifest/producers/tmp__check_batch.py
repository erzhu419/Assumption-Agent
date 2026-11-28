import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ids=np.arange(len(tr['y'])); scale,bias=m._calibrate(s['panel_a']); print('calibration',scale,bias)
W=m._fit(m._affine(s['panel_b']['u']),s['panel_b']['v']); q=m._affine((tr['x']-bias)/scale)@W
folds=[(f'mod4-{i}',ids%4==i) for i in range(4)]+[('last16',ids>=56)]
for i in range(4):
 cut=np.quantile(np.abs(q[:,i]),.75); folds.append((f'tail-{i}',abs(q[:,i])>=cut))
for label,k in folds:
 st=dict(s,train={**tr,**{a:tr[a][~k] for a in ['x','y','traj']}},test={'x':tr['x'][k]}); out=m.solve_batch(st)
 e=abs(out['state_pred']-tr['traj'][k]).mean(0)
 print(label,'y / all-state / worst horizon / terminal', np.round([abs(out['y_hat']-tr['y'][k]).mean(),e.mean(),e.max(),e[-1]],6))
