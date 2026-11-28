import importlib.util, pickle, numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; u=s['train']; x=u['x']; N=len(x); idx=np.arange(N)
def check(i,j):
 d=dict(s); d['train']={k:v[i] if k in ('x','y','traj') else v for k,v in u.items()}; d['test']={'x':x[j]}; v=m.solve_batch(d); a=abs(u['y'][j]-v['y_hat']); b=abs(u['traj'][j]-v['state_pred']); return [a.mean(),b.mean(),b.mean(0).max()]
print('official',check(idx[:52],idx[52:])); rng=np.random.default_rng(781); perms=[rng.permutation(idx) for _ in range(30)]; print('random worst',np.max([check(i[:52],i[52:]) for i in perms],axis=0)); stress=[]
for k,w in enumerate([*x.T,np.linalg.norm(x,axis=1)]):
 i=np.argsort(w); stress.extend([check(i[:52],i[52:]), check(i[20:],i[:20])]);
print('coordinate tail worst',np.max(stress,axis=0)); print(np.round(stress,4))
