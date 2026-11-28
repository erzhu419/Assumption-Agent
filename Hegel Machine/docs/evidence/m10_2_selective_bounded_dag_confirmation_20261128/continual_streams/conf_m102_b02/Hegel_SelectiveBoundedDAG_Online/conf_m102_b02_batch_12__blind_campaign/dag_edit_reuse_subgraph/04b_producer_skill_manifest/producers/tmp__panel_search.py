import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
F={
'erf3':(lambda q,p: erf(np.exp(p[0])*q+p[1]*q**3)+p[2]*q**3+p[3]*q, [[.63,4,-.62,-2.13],[0,0,0,0]]),
'rat3':(lambda q,p: q*abs(q)/(1+np.exp(p[0])*abs(q))+p[1]*q**3+p[2]*q, [[-.42,-.02,-1.08],[0,0,0]]),
'poly':(lambda q,p: q+p[0]*q**2+p[1]*q**3+p[2]*q**4+p[3]*q**5, [[0,0,0,0]]),
'abs':(lambda q,p: q+p[0]*q*abs(q)+p[1]*q**3+p[2]*q**2+p[3]*abs(q), [[0,0,0,0]]),
'rat1':(lambda q,p: q/(1+np.exp(p[0])*abs(q))+p[1]*q+p[2]*q**2, [[0,0,0]]),
'rat2':(lambda q,p: q/(1+np.exp(p[0])*q*q)+p[1]*q+p[2]*q**2, [[0,0,0]]),
'rat2even':(lambda q,p: q*q/(1+np.exp(p[0])*q*q)+p[1]*q+p[2]*q**2, [[0,0,0]]),
'rat2mix':(lambda q,p: (q+p[1]*q*q)/(1+np.exp(p[0])*q*q)+p[2]*q+p[3]*q*q, [[0,0,0,0]]),
'rat1mix':(lambda q,p: (q+p[1]*q*q+p[4]*q*abs(q))/(1+np.exp(p[0])*abs(q))+p[2]*q+p[3]*q*q, [[0,0,0,0,0]]),
'sin':(lambda q,p: np.sin(np.exp(p[0])*q)+p[1]*q+p[2]*q**2+p[3]*q**3, [[0,0,0,0],[.7,0,0,0]]),
'sincos':(lambda q,p: np.sin(np.exp(p[0])*q)+p[1]*(np.cos(np.exp(p[0])*q)-1)+p[2]*q+p[3]*q**2, [[0,0,0,0],[.7,0,0,0]]),
'erf':(lambda q,p: erf(np.exp(p[0])*q)+p[1]*q+p[2]*q**2+p[3]*q**3, [[0,0,0,0]]),
'tanh':(lambda q,p: np.tanh(np.exp(p[0])*q+p[1])+p[2]*q+p[3]*q**2+p[4]*q**3, [[0,0,0,0,0]]),
'exp':(lambda q,p: np.expm1(p[0]*q)+p[1]*q+p[2]*q**2, [[.5,0,0],[-.5,0,0]]),
}
def fit(u,v,name,keep=None):
 f,starts=F[name]; keep=slice(None) if keep is None else keep
 def d(q,p): return np.column_stack((np.ones(len(q)),f(q,p)))
 def profile(p):
  a=d(u[keep],p); w=np.linalg.lstsq(a,v[keep],rcond=None)[0];return (a@w-v[keep]).ravel(),w
 opts=[]
 for start in starts:
  p=least_squares(lambda p:profile(p)[0],start,max_nfev=120,bounds=(-15,15)).x
  res,w=profile(p);opts.append((np.mean(res**2),p,w))
 err,p,w=min(opts,key=lambda e:e[0]);return err**.5,p,lambda z:d(z,p)@w
if __name__=='__main__':
 for key,tgt in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[key]['u'],s[key][tgt];print(key,tgt)
  for n in F:
   e,p,fun=fit(u,v,n); print(n,round(e,8),p.round(5))
