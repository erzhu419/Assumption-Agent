from explore import *
from itertools import product
B=bv; C=(u*np.roll(B,1,axis=1)).sum(1)
N=len(y); base=np.arange(N); rng=np.random.default_rng(87)
folds=[(base[:52],base[52:])]+[(p[:52],p[52:]) for p in [rng.permutation(N) for i in range(30)]]
L={'atan':np.arctan,'tanh':np.tanh,'asinh':np.arcsinh,'erf':__import__('scipy.special',fromlist=['erf']).erf,'soft':lambda z:z/np.sqrt(1+z*z),'cubic':lambda z:z-z**3,'den':lambda z:z/(1+z*z)}
rows=[]
for (kind, fn), mode, offsets in product(L.items(), ['linear','square','saturate','scale','linearB'], [True,False]):
 def pred(p, J):
  x=B[J]@p[:4]; c=C[J]; a,cs,di,do,extra=p[4:]
  v=fn(x+(di if offsets else 0))
  if mode=='square': v=a*v+extra*c*c
  elif mode=='saturate': v=a*v+extra*(np.tanh(c)-c)
  elif mode=='scale': v=(a+extra*c)*v
  elif mode=='linearB': v=a*v+extra*x
  else:v=a*v
  return v+cs*c+(do if offsets else 0)
 err=[]; fit_err=[]
 for I,J in folds:
  p=np.r_[[-.5,-.5,.65,.4],2.2,.72,0.,0.,0.]; res=least_squares(lambda p:pred(p,I)-y[I],p,max_nfev=140)
  err.append(np.mean(abs(pred(res.x,J)-y[J])));fit_err.append(np.mean(abs(res.fun)))
  if J is folds[0][1]: first=res.x
 rows.append((np.mean(err),max(err),err[0],np.mean(fit_err),kind,mode,offsets,np.round(first,3)))
for row in sorted(rows,key=lambda r:r[0])[:22]:print(row)
