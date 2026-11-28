exec(open('/tmp/finalcv08.py').read().split('E=[]')[0]); from scipy.optimize import minimize_scalar
from itertools import combinations
base,Q=m._features(z,w); t=t['y']; d=z[:,3]; I=np.arange(n); rows=[]
for loc in ['plain','lin','all','int']:
 for extra in ['none','bias','outbias']:
  for k in range(0,5):
   zetas=[]
   for cols in combinations(range(10),k):
    def fit(i,j,ret=False):
     def fn(q,out=False):
      f=(1+q*d*d); A=np.c_[base[:,:5]/f[:,None] if loc in ['lin','all'] else base[:,:5],base[:,5:]/f[:,None] if loc in ['int','all'] else base[:,5:],Q[:,cols]/f[:,None] if loc in ['int','all'] else Q[:,cols]]
      if extra=='bias': A=np.c_[A,d,d/f]
      if extra=='outbias': A=np.c_[A,d]
      c=np.linalg.lstsq(A[i],t[i],rcond=None)[0]; r=A[i]@c-t[i];return (A[j]@c,c,q) if out else r@r
     q=minimize_scalar(fn,bounds=(.03,.65),method='bounded').x if loc!='plain' else .2
     return fn(q,ret)
    e=fit(I,I); zetas.append((e,cols,fit))
   for err,cols,fn in sorted(zetas,key=lambda e:e[0])[:1]:
    # Capture the selected subset before running diagnostic fits.
    yh,c,q=fn(I,I,True); ee=[np.mean(np.abs(fn(i,j,True)[0]-t[j])) for i,j in S]; rows.append((np.mean(ee[:10]),max(ee[10:]),err,loc,extra,cols,ee[0],q,np.round(c,3)))
print('sparse diagnostics');print(*sorted(rows)[:12],sep='\n')
