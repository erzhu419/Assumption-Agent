import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from transport_refine import *
from itertools import product
from scipy.optimize import least_squares

def axes_for(p,kind):
 out=[]
 for t,B in enumerate(bases):
  d=('x','y')[t]; a=B.toarray(); n=len(a); k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(n-1)
  if kind in ('diffusion','both'):
   # Interpolate from face diffusion to its adjacent cell's value.
   z=p[-2+t]*(k[:-1]-k[1:])/(2*h*h); i=np.arange(n-1)
   a[i,i+1]-=z; a[i+1,i]+=z; a[i,i]+=z; a[i+1,i+1]-=z
  if kind in ('wall-row','wall-column','both'):
   if kind=='wall-column': a[1,0]+=p[4+2*t]; a[-2,-1]+=p[5+2*t]
   else: a[0,1]+=p[4+2*t]; a[-1,-2]+=p[5+2*t]
  out.append(sp.csc_matrix(a))
 return out

def rms(a): return np.sqrt(np.mean(a*a))
for kind,n in [('diag',4),('diffusion',6),('wall-row',8),('wall-column',8),('both',10)]:
 def residual(p): return pred(p[:4],axes_for(p,kind))-Y
 init=np.r_[r0,np.zeros(n-4)]; fit=least_squares(lambda p:residual(p).ravel(),init,gtol=1e-12)
 oof=np.empty_like(Y)
 for test in range(3):
  keep=np.arange(len(Y))//len(idx)!=test
  q=least_squares(lambda p:residual(p)[keep].ravel(),init,gtol=1e-12)
  oof[~keep]=residual(q.x)[~keep]
 print(kind,'params',np.round(fit.x,6),'fit',rms(fit.fun),'worst held shift',max(map(rms,np.split(oof,3))))
print('observation range',s['y_obs_raw'].min(),s['y_obs_raw'].max())
