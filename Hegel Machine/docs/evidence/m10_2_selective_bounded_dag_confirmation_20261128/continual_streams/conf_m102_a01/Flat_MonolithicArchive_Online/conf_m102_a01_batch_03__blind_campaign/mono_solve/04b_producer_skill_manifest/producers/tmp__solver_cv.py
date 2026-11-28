import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task');import batch_solver
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['y']); rng=np.random.default_rng(2)
rows=[np.arange(n)]+[rng.permutation(n) for _ in range(7)]; results=[]; ts=[]
for p in rows:
 a,b=p[:52],p[52:]; d={**s,'train':{**t,**{k:t[k][a] for k in ('x','y','traj')}},'test':{'x':t['x'][b]}}; out=batch_solver.solve_batch(d); ey=out['y_hat']-t['y'][b]; et=out['state_pred']-t['traj'][b]; results.append(np.mean(abs(ey))); ts.append(np.mean(abs(et),0));print('split',len(results)-1,'y',results[-1],'state',ts[-1].mean(),'last',ts[-1][-1],flush=True)
print('worst scalar',max(results),'mean',np.mean(results),'worst trajectory horizon',np.max(ts))
