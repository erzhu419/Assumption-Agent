import pickle, importlib.util, numpy as np
path='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('candidate',path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x,y,z=[s['train'][k] for k in ('x','y','traj')]; n=len(y)
cal=m._calibrate(s['panel_a']); u,v=[s['panel_a'][k] for k in ('u','v')]; p,b,d=cal
print('calibration',p, 'residual norm quantiles',np.quantile(np.linalg.norm(b+d*u+p[2]*np.tanh(p[0]*u)+p[3]*np.sin(p[1]*np.roll(u,2,1))-v,axis=1),[0,.5,.8,.9,1]));
h=m._operator(s['panel_b'])(m._inverse(x,cal)); idx=np.argsort(np.linalg.norm(h,axis=1)); rng=np.random.default_rng(193)
folds=[('public',np.arange(52,n)),('operator tail',idx[-20:]),('raw tail',np.argsort(np.linalg.norm(x,axis=1))[-20:])]+[(f'random {i}',rng.permutation(n)[:20]) for i in range(8)]
for name,j in folds:
 i=np.setdiff1d(np.arange(n),j); zz=z[i,-1]; ss={**s,'train':{'x':x[i],'y':y[i],'traj':z[i],'summary':np.array([y[i].mean(),y[i].std(),zz.mean(),zz.std()])},'test':{'x':x[j]}}
 out=m.solve_batch(ss); e=np.abs(out['state_pred']-z[j]); print(name, 'scalar',round(np.mean(np.abs(out['y_hat']-y[j])),6),'state',round(e.mean(),6),'worst time',round(e.mean(0).max(),6))
print('scalar coefficients',m._ls(m._quadratic(h),y).round(5));
