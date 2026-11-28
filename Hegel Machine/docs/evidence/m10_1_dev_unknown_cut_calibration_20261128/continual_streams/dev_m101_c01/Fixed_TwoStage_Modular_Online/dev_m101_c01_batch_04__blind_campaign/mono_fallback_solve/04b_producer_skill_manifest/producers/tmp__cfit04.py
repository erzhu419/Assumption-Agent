import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; P=s['panel_c']; u=P['u']; V=np.c_[P['v1'],P['v2']]
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,k):return np.column_stack([np.prod(x[:,i],axis=1) for j in range(k+1) for i in cr(range(x.shape[1]),j)])
funcs={'rat':lambda x,k:x/(1+k*x*x),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'cos':lambda x,k:np.cos(k*x)-1,'exp':lambda x,k:np.expm1(-k*x*x),'sqrat':lambda x,k:x*x/(1+k*x*x),'soft':lambda x,k:x/np.sqrt(1+k*x*x)}
def fit_c(u,v,name,mode):
 fun=funcs[name]; n=u.shape[1]; shared=mode%2==0; linear=mode<2 or name in ('cos','exp','sqrat'); shifted=mode>=4; m=1 if shared else n
 def H(x,p):return np.c_[aff(x) if linear else np.ones(len(x)),fun(x+(p[m:] if shifted else 0),np.exp(p[:m]))]
 def funres(p,ret=False):
  h=H(u,p); w=np.linalg.lstsq(h,v,rcond=None)[0]; return w if ret else (h@w-v).ravel()
 best=min((least_squares(funres,np.r_[np.repeat(z,m),np.zeros(n if shifted else 0)],bounds=(-3,2),max_nfev=100) for z in [-1,0,.7]),key=lambda b:b.cost)
 return lambda x:H(x,best.x)@funres(best.x,True),best.x
if __name__=='__main__':
 for v,names in [(V[:,:4],('rat','tanh','sin','atan','soft')),(V[:,4:],('cos','exp','sqrat'))]:
  rows=[]
  for name in names:
   for mode in range(6):
    f,p=fit_c(u[:18],v[:18],name,mode); err=np.sqrt(np.mean((f(u)-v)**2,axis=1)); rows.append((err[18:].mean(),err[:18].mean(),name,mode,p.round(4)))
  for r in sorted(rows,key=lambda r:r[0])[:12]:print(r)
 z=s['train']['traj']; x=s['train']['x']; t=s['eval_times']; U,S,B=np.linalg.svd(z,full_matrices=False); print('temporal SVD second', B[1]); R=z-(z@B[0])[:,None]*B[0];
 for name,d in [('lin',aff(x)),('quad',poly(x,2)),('lin_amp',np.c_[aff(x),(z@B[0])[:,None]*aff(x)])]:
  w=np.linalg.lstsq(d[:56],R[:56],rcond=None)[0]; print(name,'resid predicted by time', np.sqrt(np.mean((R[56:]-d[56:]@w)**2,axis=0)), 'coef lin',w[1:5].round(5))
 print('resid baseline by time', np.sqrt(np.mean(R[56:]**2,axis=0)))
