import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def I(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def mae(x):return np.mean(abs(x))
def fit(f,u,v):
 a=f(u); r=a@ls(a,v)-v; h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1);return np.sqrt(np.mean(r*r)),mae(r/(1-h[:,None]))
u,v=[s['panel_a'][k] for k in ('u','v')];keep=np.linalg.norm(v,axis=1)<5; print('A dropped',np.flatnonzero(~keep));u,v=u[keep],v[keep]
from itertools import combinations_with_replacement
pows=[p for d in (1,2,3,4) for p in combinations_with_replacement(range(4),d)]
X=lambda u,p:I(np.column_stack([np.prod(u[:,i],axis=1) for i in p]));
for deg in range(1,5):
 p=[p for p in pows if len(p)<=deg];a=X(u,p);c=ls(a,v);print('A degree',deg,mae(a@c-v));
 if deg in (2,3):
  for j in range(4): print(j,[(p[i],round(c[i+1,j],4)) for i in range(len(p)) if abs(c[i+1,j])>.012], 'offset',c[0,j])
F={'sin':lambda u,k:np.sin(k*u),'tanh':lambda u,k:np.tanh(k*u),'asinh':lambda u,k:np.arcsinh(k*u),'rat':lambda u,k:u/(1+k*u*u),'soft':lambda u,k:u/(1+k*abs(u)),'atan':lambda u,k:np.arctan(k*u),'erf':lambda u,k:__import__('scipy').special.erf(k*u),'soft2':lambda u,k:u/np.sqrt(1+k*u*u),'cos':lambda u,k:1-np.cos(k*u),'sq':lambda u,k:u*u,'cube':lambda u,k:u**3,'abs':lambda u,k:abs(u),'expe':lambda u,k:1-np.exp(-k*u*u)}
for pan,out in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[pan]['u'],s[pan][out];res=[]; print(pan,out)
 for name,g in F.items():
  for coupled in [False,True]:
   fun=(lambda u,p:I(u+p[1]*g(u,p[0]))) if coupled else (lambda u,p:I(np.c_[u,g(u,p[0])]))
   def r(p):
    a=fun(u,p);return (a@ls(a,v)-v).ravel()
   cand=[least_squares(r,[k,-.4] if coupled else [k],bounds=([.03,-10],[10,10]) if coupled else ([.03],[10]),max_nfev=150) for k in [.4,1.25,2.15,3.5,5.]]; b=min(cand,key=lambda b:b.cost);res.append((fit(lambda u:fun(u,b.x),u,v),name,coupled,b.x))
 for z in sorted(res,key=lambda z:z[0][1])[:12]: print(z)
 c=ls(I(np.c_[u,np.sin(2.15*u),np.tanh(1.25*u),u*u]),v);print('mixed coefficients',np.round(c,3))
