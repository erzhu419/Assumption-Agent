import numpy as np,pickle,runpy
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; Y=np.c_[tr['y'],tr['traj']]; n=len(Y); ix=np.arange(n); rng=np.random.default_rng(949)
F=m['_design'](s,tr['x']); L=F[:,:29]; gh=F[:,29:].reshape(n,4,4); diag=np.trace(gh,axis1=1,axis2=2); D=np.c_[L,diag]; E=np.c_[D,gh[:,0,1]]
sets=[ix[52:]]+[ix[i:i+20] for i in [0,13,26,39]]+[rng.choice(n,20,False) for _ in range(30)]+[np.argsort(np.linalg.norm(tr['x'],axis=1))[-20:]]
def exp_basis(r): return -np.expm1(-r[:,None]*t)
def exp_res(r,A):
 B=exp_basis(r); return (m['_ls'](B.T,A.T).T@B-A).ravel()
for nfit in [52,72]:
 A=Y[:nfit,1:]; r=least_squares(exp_res,[.2,.7,1.8],args=(A,),bounds=(.005,15)).x; print('global rates',r,'rmse',np.sqrt(np.mean(exp_res(r,A)**2)))
 B=m['_ls'](E[:nfit],A); print('compact coef curves',np.round(B[[5,6,7,8,9,10,11,12,-2,-1]],4));
def trial(yfeatures,sfeatures,k):
 out=[]
 for v in sets:
  p=np.setdiff1d(ix,v); A=Y[p,1:]; by=np.linalg.svd(A,full_matrices=False)[2][:3] if k==3 else np.eye(len(t));
  if k==5:
   r=least_squares(exp_res,[.2,.7,1.8],args=(A,),bounds=(.005,15),max_nfev=150).x; by=np.linalg.qr(exp_basis(r).T)[0].T
  y=yfeatures[v]@m['_ls'](yfeatures[p],Y[p,0]); a=sfeatures[v]@m['_ls'](sfeatures[p],A@by.T)@by; err=np.abs(a-Y[v,1:]); out.append([np.mean(abs(y-Y[v,0])),err.mean(),err.mean(0).max()]);
 o=np.array(out); print('model',yfeatures.shape[1],sfeatures.shape[1],k,'required',o[0],'worst',o.max(0),'boundary',o[-1])
for a,b in [(F,F),(D,E),(D,F),(D,np.delete(E,[1,2,3,4],axis=1)),(np.delete(D,[1,2,3,4],axis=1),E)]:
 for k in [3,5,8]:trial(a,b,k)
