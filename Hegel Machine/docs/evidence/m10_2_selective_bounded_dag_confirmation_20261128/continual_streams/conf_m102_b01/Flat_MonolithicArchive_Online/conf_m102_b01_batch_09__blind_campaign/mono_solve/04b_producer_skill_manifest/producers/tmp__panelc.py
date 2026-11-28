from explore import s,aff,ls
import numpy as np
from scipy.optimize import minimize_scalar
c=s['panel_c'];u=c['u'];
def cv(d,v):
 pi=np.linalg.pinv(d);return np.mean(((d@(pi@v)-v)/(1-(d*pi.T).sum(1))[:,None])**2)**.5
for k in ['v1','v2']:
 v=c[k]; scores=[]
 fs={'cubic':lambda x:np.sinh(np.arcsinh(x)/3),'arcsinh':np.arcsinh,'atan':np.arctan,'tanh':np.tanh,'swish':lambda x:x/(1+np.exp(-x)),'log':lambda x:np.log1p(x*x),'exp':lambda x:-np.expm1(-x*x),'soft':lambda x:x/np.sqrt(1+x*x),'rat':lambda x:x/(1+x*x)}
 for nm,f in fs.items():
  for mode in ['f','f+u','u+f','u-f']:
   def d(x,p):
    q=f(p*x)
    return aff({'f':q,'f+u':np.c_[q,x],'u+f':x+q,'u-f':x-q}[mode])
   p=minimize_scalar(lambda p:cv(d(u,p),v),bounds=(.15,12),method='bounded').x;scores.append((cv(d(u,p),v),nm,mode,p))
 print(k,sorted(scores)[:12])
