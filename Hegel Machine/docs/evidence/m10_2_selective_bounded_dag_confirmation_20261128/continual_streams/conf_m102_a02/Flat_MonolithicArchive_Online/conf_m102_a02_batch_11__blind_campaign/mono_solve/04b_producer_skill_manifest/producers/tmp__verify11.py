import pickle, numpy as np, importlib.util
p='/holdout_task/'
sp=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj'];
x=m._inverse(tr['x'],m._calibrate(s['panel_a']));print('state singular values',np.linalg.svd(T,compute_uv=False)); print('times',s['eval_times']); print('state affine coefficients',m._ls(m._aff(x),T).round(5));
I=np.arange(len(y)); splits=[('52/20',I)]+[(f'random_{k}',np.random.default_rng(k).permutation(I)) for k in range(12)]+[(f'{name}_extreme_{h}',np.argsort(z)[::h]) for name,z in [('norm',np.linalg.norm(x,axis=1)),('ridge',x@np.array([-.52,-.15,1.,-.12]))] for h in [1,-1]]
for nm,i in splits:
 a,b=i[:52],i[52:]; q=dict(s,train=dict(x=tr['x'][a],y=y[a],traj=T[a],summary=np.array([y[a].mean(),y[a].std(),T[a,-1].mean(),T[a,-1].std()])),test={'x':tr['x'][b]}); r=m.solve_batch(q); e=abs(r['y_hat']-y[b]); f=abs(r['state_pred']-T[b]); print(nm, 'scalar MAE/max',e.mean(),e.max(),'traj MAE/worst time',f.mean(),f.mean(0).max())
