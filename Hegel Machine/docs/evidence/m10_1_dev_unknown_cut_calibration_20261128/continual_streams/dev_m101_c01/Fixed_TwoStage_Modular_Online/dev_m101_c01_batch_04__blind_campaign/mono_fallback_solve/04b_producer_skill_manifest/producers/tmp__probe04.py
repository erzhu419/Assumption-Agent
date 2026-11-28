import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; q=s['test']['x']; n=len(x)
def aff(x): return np.c_[np.ones(len(x)),x]
def fitpanel(p,key):
 u=aff(p['u']); v=p[key]; rng=np.random.default_rng(11)
 def score(w): return np.sum(np.sort(np.sum((u@w-v)**2,axis=1))[:int(.75*len(u))])
 w=min([np.linalg.lstsq(u,v,rcond=None)[0]]+[np.linalg.lstsq(u[i],v[i],rcond=None)[0] for i in (rng.choice(len(u),7,False) for _ in range(400))],key=score)
 for _ in range(6):
  err=np.linalg.norm(u@w-v,axis=1); keep=err<max(5*np.median(err),.001); w=np.linalg.lstsq(u[keep],v[keep],rcond=None)[0]
 return w,err,keep
A,err,keep=fitpanel(s['panel_a'],'v'); B,_,_=fitpanel(s['panel_b'],'v'); print('A=',A,'err',np.sort(err),'Gram',A[1:]@A[1:].T,'B',B,sep='\n')
def poly(x,k):return np.column_stack([np.prod(x[:,i],axis=1) for j in range(k+1) for i in combinations_with_replacement(range(x.shape[1]),j)])
def expbasis(r):return -np.expm1(-t*r)
def timepred(p):
 f=expbasis(p[0]); a=z@f/(f@f); return a,f
r=least_squares(lambda p:(z-np.outer(*timepred(p))).ravel(),[1.]).x; a,f=timepred(r); print('rate',r,'residual rms by time',np.mean((z-np.outer(a,f))**2,axis=0)**.5)
for i in np.argsort(abs(a))[-12:]:
 rr=least_squares(lambda p:z[i]-expbasis(p[0])*(z[i]@expbasis(p[0]))/sum(expbasis(p[0])**2),r).x[0]; print('amp/rate/raw/cal',round(a[i],4),round(rr,4),np.round(x[i],3),np.round(aff(x)[i]@A,3))
coords={'raw':x,'A':aff(x)@A,'B':aff(x)@B,'AB':aff(aff(x)@A)@B}
try:coords['invA']=(x-A[0])@np.linalg.inv(A[1:])
except np.linalg.LinAlgError:pass
for name,u in coords.items():
 print('corr',name,np.corrcoef(np.c_[y,a,u].T)[:2,2:]);
 for k in [1,2,3]:
  F=poly(u,k); W=np.linalg.lstsq(F[:56],np.c_[y,a][:56],rcond=None)[0]; e=np.abs(F@W-np.c_[y,a]);
  print('poly',name,k,e[:56].mean(0),e[56:].mean(0))
 if name!='raw' and name!='A':continue
 for Y in [y,a]:
  F=poly(u,2); T=F[:56]; yy=Y[:56]; results=[]
  for kind in ['single','rad','diag','full','rad2','mult','tanh']:
   H=aff(u) if kind in ('single','mult','tanh','rad2') else np.c_[aff(u),np.sum(u*u,axis=1)] if kind=='rad' else np.c_[aff(u),u*u] if kind=='diag' else F
   m=H.shape[1]+(kind in ('rad2','mult')); reg=np.r_[np.full(5,.0006),np.full(m-5,.015)]; R=np.sum(u*u,axis=1)
   def factor(p):
    g=H@p[:H.shape[1]]
    if kind=='rad2':return 1/(1+g*g+np.exp(p[-1])*R)
    if kind=='mult':return 1/((1+g*g)*(1+np.exp(p[-1])*R))
    if kind=='tanh':return np.tanh(g)/np.where(abs(g)>.00001,g,1e-5)+(abs(g)<=.00001)
    return 1/(1+g*g)
   def residual(p,ret=False):
    fac=factor(p); D=T*fac[:56,None]; w=np.linalg.lstsq(D,yy,rcond=None)[0]; return (F@w)*fac if ret else np.r_[D@w-yy,reg*p]
   rng=np.random.default_rng(32); starts=rng.normal(0,.3,(6,m));
   if kind in ('rad2','mult'):starts[:,-1]=-2
   sol=min([least_squares(residual,p,bounds=(-9,9),max_nfev=110) for p in starts],key=lambda v:v.cost); e=np.abs(residual(sol.x,True)-Y); results.append((kind,round(e[:56].mean(),5),round(e[56:].mean(),5),np.round(sol.x,3).tolist()))
  print('rational',name,'y' if Y is y else 'amp',results)
