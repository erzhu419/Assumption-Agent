import pickle, numpy as np
from itertools import combinations
from scipy.optimize import minimize_scalar
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; X=s['ref_rosette']; Y=s['raw_rosette']; ok=np.max(abs(Y-np.median(Y,axis=0)),axis=1)<5; X=X[ok]; Y=Y[ok]; b=s['single_bank_sweeps']; B=b['bank_id']; R=b['ref_signal']; V=b['raw_readout']; good=abs(V)<5
funcs={'tanh':np.tanh,'erf':erf,'sin':np.sin,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'root':lambda z:z/np.sqrt(1+z*z),'cubrat':lambda z:z*(1-z*z)/(1+z*z)}
def solve(A,y):
 c=np.linalg.lstsq(A,y,rcond=None)[0]; return A@c-y
for typ in ['diag','cross']:
 hits=[]
 for name,f in funcs.items():
  for odd in [(),(1,),(3,),(1,3),(1,3,5)]:
   for even in ['none','x2','f2','r2','x2f2','x2x4']:
    def design(x,k):
     z=k*x; ff=f(z); e={'none':[], 'x2':[x*x], 'f2':[ff*ff], 'r2':[x*x/(1+z*z)], 'x2f2':[x*x,ff*ff], 'x2x4':[x*x,x**4]}[even]; return np.column_stack([ff,*[x**p for p in odd],*e])
    def errs(k):
     rr=[]
     for j in range(4):
      if typ=='diag':
       sel=(B==j)&good; x=R[sel]; y=V[sel]; A=design(x,k)
      else:
       x=X[:,j^2]; y=Y[:,j]; A=np.column_stack([design(x,k),*[X[:,j]**p for p in range(1,10)]])
      rr.extend(solve(np.c_[np.ones(len(x)),A],y))
     return np.sqrt(np.mean(np.square(rr)))
    grid=np.geomspace(.25,3,12); m=np.argmin([errs(k) for k in grid]); r=minimize_scalar(errs,bounds=(grid[max(m-1,0)],grid[min(m+1,len(grid)-1)]),method='bounded',options={'xatol':1e-9}); hits.append((r.fun,name,odd,even,r.x))
 print(typ, 'best', sorted(hits)[:16]); print('parsimonious',sorted(h for h in hits if len(h[2])<=2 and h[3] in ('none','x2','f2','r2'))[:10])
