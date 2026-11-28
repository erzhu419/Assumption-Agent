import runpy,pickle,numpy as np
G=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; R=np.arange(72)
for te in [R[52:],R[:20],R[20:40],R[40:60]]+[np.random.default_rng(i).permutation(R)[:20] for i in (0,1,2)]:
 tr=np.setdiff1d(R,te); train={k:v[tr] if k!='summary' else v for k,v in a.items()}; o=G['solve_batch'](dict(s,train=train,test={'x':a['x'][te]})); u=abs(o['y_hat']-a['y'][te]); v=abs(o['state_pred']-a['traj'][te]);print(te[:3],u.mean(),v.mean(),v.mean(0).round(4),flush=True)
M,_,_=G['_spatial'](s,a['x']); k=G['_prune'](M[:52],a['y'][:52,None],.0045); print('scalar terms',k,G['_fit'](M[:52,k],a['y'][:52]).round(5))
