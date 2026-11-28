import pickle,runpy,numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; tr=t['traj']; x=t['x']; n=len(y); ix=np.arange(n)
m=runpy.run_path('/holdout_task/batch_solver.py'); z=m['_calibrate'](s['panel_a'],x); w=m['_panel_b'](s['panel_b'],z); a=m['_affine'](w); fit=m['_ls']
d=cdist(w,w,'sqeuclidean'); norm=np.linalg.norm(w-w.mean(0),axis=1); tail=np.argsort(norm)[-16:]; near=np.argsort(np.partition(d,1,axis=1)[:,1])[-16:]
splits=[('gate52',ix[52:])]+[(f'cv{i}',v) for i,v in enumerate(np.array_split(np.random.default_rng(17).permutation(n),4))]+[('high_norm',tail),('sparse',near)]
poly=[np.column_stack([np.prod(w[:,j],axis=1) for j in combinations_with_replacement(range(4),o)]) for o in (2,3,4)]
def predict(kernel, it, iv, reg):
 q=solve(kernel[np.ix_(it,it)]+reg*np.eye(len(it)),np.c_[a[it],y[it]],assume_a='pos'); c=fit(a[it].T@q[:,:-1],a[it].T@q[:,-1]); return a[iv]@c+kernel[np.ix_(iv,it)]@(q[:,-1]-q[:,:-1]@c)
g=np.exp(-.5*d); p=sum(f@f.T*weight for f,weight in zip(poly,(.5,1.,.2))); p*=np.linalg.norm(g)/np.linalg.norm(p)
for name,k,reg in [('gaussian',g,.005),('mixed',g+.3*p,.005),('polynomial',p,.004),('matern',(1+np.sqrt(3*d))*np.exp(-np.sqrt(3*d)),.005)]:
 print(name,[(name2,round(np.abs(predict(k,np.setdiff1d(ix,iv),iv,reg)-y[iv]).mean(),6)) for name2,iv in splits])
for name,iv in splits:
 it=np.setdiff1d(ix,iv); train={key:t[key][it] for key in ('x','y','traj')}; end=train['traj'][:,-1]; yy=train['y']; train['summary']=np.array([yy.mean(),yy.std(),end.mean(),end.std()]); o=m['solve_batch'](dict(s,train=train,test={'x':x[iv]})); e=np.abs(o['state_pred']-tr[iv]); u=np.abs(o['y_hat']-y[iv]); print(name,'runtime scalar',round(u.mean(),6),'trajectory',round(e.mean(),6),'horizon_max',round(e.mean(0).max(),6),'scalar_max',round(u.max(),6))
 if name=='gate52': assert u.mean()<=.038 and e.mean()<=.038
