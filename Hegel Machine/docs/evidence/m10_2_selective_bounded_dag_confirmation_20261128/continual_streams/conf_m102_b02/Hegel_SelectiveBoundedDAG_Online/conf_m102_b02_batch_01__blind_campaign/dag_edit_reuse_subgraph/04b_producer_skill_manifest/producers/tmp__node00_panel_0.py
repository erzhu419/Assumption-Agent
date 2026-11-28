from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.optimize import brentq
"""Robustly recover latent input coordinates from panel A."""
import numpy as np


def _latent(study):
    panel = study['panel_a']
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    d = np.column_stack((np.ones(len(u)), u))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        coef = np.linalg.lstsq(d[keep], v[keep], rcond=None)[0]
        e = np.linalg.norm(d @ coef - v, axis=1)
        nxt = e <= max(0.001, 3 * np.median(e))
        if np.array_equal(keep, nxt) or nxt.sum() < d.shape[1]:
            break
        keep = nxt
    coef = np.linalg.lstsq(d[keep], v[keep], rcond=None)[0]
    x, xt = (np.asarray(study[k]['x'], dtype=float) for k in ('train', 'test'))
    z = np.linalg.solve(coef[1:].T, (np.vstack((x, xt)) - coef[0]).T).T
    return z, len(x)




# Panel-only alternatives: jointly fit a common distortion law and its mixing matrix.
laws={'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'cos':lambda x,k:np.cos(k*x)-1,'sin':lambda x,k:np.sin(k*x)}
def panels(s):
 out={}
 for nm,kk,ky in [('B','panel_b','v'),('C1','panel_c','v1'),('C2','panel_c','v2')]:
  p=s[kk]; U,V=p['u'],p[ky]; f=laws[{'B':'rat','C1':'tanh','C2':'cos'}[nm]]
  def fit(k):
   F=np.c_[np.ones(len(U)),U,f(U,k)]; C=np.linalg.lstsq(F,V,rcond=None)[0];return np.mean((F@C-V)**2),C
  ks=np.linspace(.2,4,48);i=np.argmin([fit(k)[0] for k in ks]);k=minimize_scalar(lambda k:fit(k)[0],bounds=(ks[max(0,i-1)],ks[min(i+1,len(ks)-1)]),method='bounded').x
  _,C=fit(k); A=C[1:5]; L,_,R=np.linalg.svd(A); M=L@R; b=C[0]; W=C[5:]; r=np.sum(A*W)/np.sum(A*A)
  def inv(t,A=A,b=b,k=k,r=r,f=f):
   q=np.linalg.solve(A.T,(t-b).T).T; grid=np.linspace(-4,4,4097); val=grid+r*f(grid,k)
   ix=np.argsort(val,kind='stable'); return np.interp(q,val[ix],grid[ix])
  out.update({nm:lambda t,A=A,b=b,k=k,W=W,f=f:t@A+f(t,k)@W+b,nm+'i':inv,nm+'L':lambda t,A=A,b=b:t@A+b,nm+'R':lambda t,M=M:t@M,nm+'T':lambda t,M=M:t@M.T,nm+'pre':lambda t,r=r,k=k,f=f:t+r*f(t,k),nm+'post':lambda t,r=r,k=k,f=f,M=M:(t@M+r*f(t@M,k))@M.T})
 out['I']=lambda t:t
 return out

def source(s,mode):
 z,n=_latent(s); p=s['train']; q=s['test']; j=mode[0]
 if j=='latent':return z,n
 if j=='x':return np.vstack([p['x'],q['x']]),n
 if j in p:
  v=np.asarray(p[j],float)
  if v.shape==(z.shape[1],):
   if mode[1]=='offset':z=z-v
   elif mode[1]=='add':z=z+v
   elif mode[1]=='scale':z=z*np.maximum(abs(v),.02)
   elif mode[1]=='divide':z=z/np.maximum(abs(v),.02)
 return z,n


def run_node(inputs: dict) -> dict:
    study=inputs["__input__"]
    q,n=source(study,('latent', 'I'))
    f=panels(study)
    q=f['C1i'](f['C1L'](q))
    q=q*(0.13/max(.0001,np.mean(q[:n].std(0))))
    return {"z":q[:n].copy(),"zt":q[n:].copy()}
