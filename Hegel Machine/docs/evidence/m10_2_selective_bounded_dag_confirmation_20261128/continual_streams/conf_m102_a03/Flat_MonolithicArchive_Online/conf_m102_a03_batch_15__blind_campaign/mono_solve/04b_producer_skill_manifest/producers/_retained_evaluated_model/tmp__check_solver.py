import pickle,runpy,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; rng=np.random.default_rng(1538); n=len(tr['x'])
for j in range(31):
    i=np.arange(n) if not j else rng.permutation(n); a,b=i[:52],i[52:]; y=tr['y'][a]; p=tr['traj'][a]; q=dict(s,train=dict(x=tr['x'][a],y=y,traj=p,summary=np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()])),test={'x':tr['x'][b]}); r=m['solve_batch'](q)
    d=np.abs(r['y_hat']-tr['y'][b]); e=np.abs(r['state_pred']-tr['traj'][b]); print(j,d.mean(),e.mean(),d.max(),e.mean(0).max())
z=m['_decode'](tr['x'],m['_calibration'](s['panel_a'])); b=m['_operator'](s['panel_b'],z); h,q=m['_readouts'](b); print('scalar coefficients',m['_fit'](q[:52],tr['y'][:52])); print('spatial coefficients',m['_fit'](h[:52],tr['traj'][:52])); print('condition',np.linalg.cond(q[:52]))
