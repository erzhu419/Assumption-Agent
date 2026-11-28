import numpy as np, pickle, runpy
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['x']); ids=np.arange(n); rng=np.random.default_rng(42)
orders=[ids,ids[::-1]]+[rng.permutation(n) for _ in range(25)]
for z in [t['x'],m['_calibrate'](s['panel_a'],t['x'])]:
 for order in [np.argsort(np.linalg.norm(z,axis=1))]+[np.argsort(z[:,j]) for j in range(z.shape[1])]: orders.extend([order,order[::-1]])
E=[]
for o in orders:
 a,b=o[:52],o[52:]; v=m['solve_batch'](dict(s,train={k:(val[a] if k!='summary' else val) for k,val in t.items()},test={'x':t['x'][b]})); ey=np.abs(v['y_hat']-t['y'][b]); et=np.abs(v['state_pred']-t['traj'][b]); E.append(np.r_[ey.mean(),et.mean(),et.mean(0),ey.max(),et.max()]);
E=np.array(E);print('prescribed',E[0]);print('random maxima',E[:27].max(0));print('extreme extrapolation maxima',E[27:].max(0));print('worst indices',E.argmax(0));
z=m['_calibrate'](s['panel_a'],t['x']); b,c=m['_branches'](s['panel_c'],z); q=m['_modes'](s['eval_times'],t['traj']); F=m['_aff'](np.c_[b,c,(b[:,:,None]*c[:,None,:]).reshape(n,-1)]); Y=np.c_[t['y'],m['_ls'](q,t['traj'].T).T]; fit=m['_ls'](F,Y); print('coefs',fit.round(4)); print('condition',np.linalg.cond(F)); print('simple y coefs',m['_ls'](m['_aff'](np.c_[b,c,np.sum(b*c,axis=1)]),t['y']));
