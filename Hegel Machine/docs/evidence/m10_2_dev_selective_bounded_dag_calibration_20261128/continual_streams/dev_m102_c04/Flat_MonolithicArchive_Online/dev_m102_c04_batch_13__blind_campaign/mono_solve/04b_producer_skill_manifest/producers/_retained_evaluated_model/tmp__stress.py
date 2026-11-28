import pickle, numpy as np, runpy
from scipy.optimize import least_squares, minimize_scalar
m=runpy.run_path('/holdout_task/batch_solver.py'); A=m['_aff']; L=m['_ls']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; n=len(T['x']); y=T['y']; q=T['traj']; tt=s['eval_times']; z=m['_calibrate'](s['panel_a'],T['x']); p,r=m['_operators'](s['panel_c'],z); common=A(p,r,np.sum(p*r,axis=1)); cross=(p[:,:,None]*r[:,None,:]).reshape(n,16); dense=A(p,r,cross); idx=np.arange(n); rng=np.random.default_rng(90827)
splits=[('public',idx[52:])]+[(f'corner{j}',np.argsort(v)[-20:]) for j,v in enumerate([*z.T,*(-z).T,np.linalg.norm(z,axis=1)])]+[(f'random{j}',rng.permutation(n)[52:]) for j in range(20)]
for name in ('current','compact','compact_sparse','compact_noint'):
 results=[]
 for label,te in splits:
  tr=np.setdiff1d(idx,te); ay=common[te]@L(common[tr],y[tr]); t=m['_time_basis'](tt,q[tr]); amps=L(t,q[tr].T).T
  if name=='current': h=(dense[te]@L(dense[tr],amps))@t.T
  else:
   cand=[]
   for j in (range(16) if name=='compact_sparse' else [1]):
    fast=np.c_[common,cross[:,j]] if name!='compact_noint' else common; slow=A(r,cross[:,j]);
    def F(k,ind):
     b=-np.expm1(-tt[:,None]*k); return np.concatenate((fast[ind,:,None]*b[:,0],slow[ind,:,None]*b[:,1]),1).transpose(0,2,1).reshape(len(ind)*len(tt),-1)
    def res(k):
     f=F(k,tr); return f@L(f,q[tr].ravel())-q[tr].ravel()
    k=least_squares(res,[1.8,.43],bounds=([.8,.03],[8.,.79])).x; f=F(k,tr); h0=(F(k,te)@L(f,q[tr].ravel())).reshape(len(te),-1); cand.append((np.mean(res(k)**2),j,k,h0))
   _,j,k,h=min(cand,key=lambda v:v[0]);
  e=np.abs(h-q[te]); results.append([np.mean(abs(ay-y[te])),e.mean(),e.mean(0).max()]);
  if label=='public': print(name,label,np.round(results[-1],7), ('j, rates, traj fit RMSE',j,k,np.sqrt(min(v[0] for v in cand))) if name!='current' else '')
 v=np.array(results); print(name,'mean',v.mean(0),'worst',v.max(0),'worst slice',[splits[i][0] for i in v.argmax(0)])
