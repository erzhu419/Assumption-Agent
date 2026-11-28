import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,t=[tr[k] for k in ('x','y','traj')]; z=m._calibrate(s['panel_a'],x); g=m._panel(s['panel_b'],z); sp=np.maximum(abs(z).max(1),abs(g).max(1)); n=len(x)
splits=[np.arange(n)]+[np.random.default_rng(i).permutation(n) for i in range(10)]+[np.argsort(sp)]
for i,p in enumerate(splits):
 j=p[52:]; yp,tp=m._predict(z,g,p[:52],j,y[p[:52]],t[p[:52]]); print(i,round(abs(y[j]-yp).mean(),6),round(abs(t[j]-tp).mean(),6),round(abs(t[j,-1]-tp[:,-1]).mean(),6))
