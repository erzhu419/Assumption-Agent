from rosette_models import X,Y,axis,basis
import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(2026); fold=np.empty(len(X),int)
for mask in (axis,~axis): fold[mask]=rng.permutation(mask.sum())%5
# Saturation with additive corrections, then test whether off-axis terms are needed.
def sat(z,k,kind):
 if kind=='t':return np.tanh(k*z)
 if kind=='r':return z/(1+(k*z)**2)
 if kind=='s':return z/np.sqrt(1+(k*z)**2)
 if kind=='a':return np.arctan(k*z)
 if kind=='e':return z*np.exp(-(k*z)**2)
def design(x,p,spec):
 ku,kv,nu,nv,cross=spec;u,v=x.T
 cols=[np.ones(len(u))]+[u**i for i in range(1,nu+1)]+[v**i for i in range(1,nv+1)]
 if ku:cols.append(sat(u,p[0],ku))
 if kv:cols.append(sat(v,p[1],kv))
 if cross:cols += [u*v,u*u*v,u*v*v]
 return np.column_stack(cols)
def fit(x,y,spec):
 def coef(p):
  D=design(x,p,spec);return np.linalg.lstsq(D,y,rcond=1e-10)[0]
 def err(p):return design(x,p,spec)@coef(p)-y
 p=least_squares(err,[1.1,.8],bounds=(.1,4),max_nfev=70).x
 return p,coef(p)
specs=[(a,b,nu,nv,c) for a in ('t','r','s','a','e') for b in ('t','r','s','a','e') for nu,nv in [(0,2),(1,2),(2,2),(3,3)] for c in (False,True)]+[(None,None,6,6,False),(None,None,9,9,False)]
for j in range(4):
 x=X[:,[j,j^2]]; y=Y[:,j]; results=[]
 for sp in specs:
  p,b=fit(x,y,sp); e=design(x,p,sp)@b-y
  results.append((np.median(abs(e)),np.sqrt(np.mean(e**2)),sp,p,b,e))
 print('CHANNEL',j)
 for med,rms,sp,p,b,e in sorted(results,key=lambda r:r[1])[:8]:
  print(sp,'med/rms/max',np.round([med,rms,max(abs(e))],7),'largest rows',np.argsort(abs(e))[-4:].tolist(),'p',p.round(5),'b',b.round(5))
  pred=np.empty_like(y)
  for k in range(5):
   tr=fold!=k; pp,bb=fit(x[tr],y[tr],sp);pred[~tr]=design(x[~tr],pp,sp)@bb
  print('OOF rms axis/mixed',*[round(np.sqrt(np.mean((pred[m]-y[m])**2)),7) for m in (axis,~axis)])
 print('extreme exact-ish raw values',[(int(i),round(y[i],6),x[i].round(4).tolist()) for i in np.where(abs(y*100-np.round(y*100))<.002)[0] if not axis[i]])
