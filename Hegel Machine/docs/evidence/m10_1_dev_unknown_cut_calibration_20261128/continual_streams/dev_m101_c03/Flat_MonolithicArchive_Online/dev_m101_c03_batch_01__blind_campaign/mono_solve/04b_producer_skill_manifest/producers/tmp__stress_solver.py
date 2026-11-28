import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; X=m._design(s,tr['x']); y=tr['y']; T=tr['traj']; ids=np.arange(len(X));
cal=m._calibrate(s['panel_a']); z=(tr['x']-cal[0])/cal[1]; radius=np.linalg.norm(z,axis=1); interaction=np.max(abs(X[:,13:]),axis=1)
splits=[('56/16',ids[56:]),('high-radius',np.argsort(radius)[-16:]),('high-interaction',np.argsort(interaction)[-16:])]+[(f'fold{k}',j) for k,j in enumerate(np.array_split(np.random.default_rng(209).permutation(len(ids)),4))];
fast=np.r_[5:15]; slow=np.r_[9:14]; full=np.r_[0:15]; ysel=np.r_[0,5:13,14];
for name,J in splits:
 I=np.setdiff1d(ids,J); K,A=m._kinetics(X[I],T[I],t)
 def pred(q,z): return X[J][:,q]@m._fit(X[I][:,q],z)
 base=pred(full,A)@K.T; narrow=np.column_stack((pred(slow,A[:,0]),pred(fast,A[:,1])))@K.T
 print(name,'y full/narrow',*[round(np.mean(abs(pred(q,y[I])-y[J])),6) for q in [full,ysel]],'traj full/narrow',*[round(np.mean(abs(p-T[J])),6) for p in [base,narrow]],'traj worst horizon full/narrow',*[round(np.max(np.mean(abs(p-T[J]),axis=0)),6) for p in [base,narrow]])
