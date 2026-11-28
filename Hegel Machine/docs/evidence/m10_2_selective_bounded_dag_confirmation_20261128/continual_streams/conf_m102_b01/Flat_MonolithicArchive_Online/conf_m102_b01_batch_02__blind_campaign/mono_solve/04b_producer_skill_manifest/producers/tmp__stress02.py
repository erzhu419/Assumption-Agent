import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; tt=tr['traj']; ix=np.arange(len(x)); h=m._latent(s,x)
coord=np.linalg.norm(x,axis=1); hf=np.linalg.norm(h,axis=1); splits=[('public',ix[52:])]+[(f'block{i}',ix[i:i+20]) for i in [0,13,26,39]]+[(n,np.argsort(v)[-20:]) for n,v in [('input_radius',coord),('latent_radius',hf),('positive_tail',tr['y']),('negative_tail',-tr['y'])]]+[(f'axis_{j}_{sign:+}',np.argsort(sign*x[:,j])[-20:]) for j in range(4) for sign in [-1,1]]+[(f'random{k}',np.random.default_rng(k).permutation(ix)[:20]) for k in range(10)]
for n,j in splits:
 i=np.setdiff1d(ix,j); a=m.solve_batch(dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':x[j]})); err=np.abs(a['state_pred']-tt[j]);print(n,round(np.mean(abs(a['y_hat']-tr['y'][j])),6),round(err.mean(),6),round(max(err.mean(axis=0)),6))
F=m._aff(h); print('spatial trajectory coefficients:\n',m._fit(F,tt)); print('scalar coefficients',m._fit(np.c_[F,h[:,0]*h[:,2],h[:,1]*h[:,3]],tr['y'])); c=s['panel_c']; C=m._aff(m._products(c['u'])); print('panel C sparse LOO MAE',np.mean(abs((c['v1']-C@m._fit(C,c['v1']))/(1-np.sum(C*np.linalg.pinv(C).T,axis=1))[:,None])))
