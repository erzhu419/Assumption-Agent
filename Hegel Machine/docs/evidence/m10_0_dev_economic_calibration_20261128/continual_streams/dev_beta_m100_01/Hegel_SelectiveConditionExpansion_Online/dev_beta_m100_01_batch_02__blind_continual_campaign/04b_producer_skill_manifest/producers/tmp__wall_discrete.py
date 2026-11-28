from wall_probe import *
from itertools import product
from functools import lru_cache

def ax(k,v,h,r,kind):
 a=axis(k,v,h,r,np.array([[1.,1.],[1.,1.]])).tolil(); n=len(k)
 for t,b,j in [(0,n-1,1),(n-1,0,n-2)]:
  if kind in ('open','reflect'):
   if kind=='reflect': a[t,t]-=r/h
  elif kind=='pin':
   a[t,j]=0; a[t,t]=1/h**2
  else:
   f=k[t] if kind=='periodic-local' else (k[t]+k[j])/2
   g=k[t] if kind=='periodic-local' else (k[t]+k[b])/2
   if kind=='periodic-upwind':
    f+=h*max(v[t] if t==0 else -v[t],0); g+=h*max(-v[t] if t==0 else v[t],0)
   a[t,j]=-f/h**2; a[t,b]=-g/h**2; a[t,t]=(f+g)/h**2
 return a.tocsc()

@lru_cache(None)
def pred(x,q):
 A=ax(s['kappa_x'],s['vel_x'],s['Lx']/(nx-1),s['robin_x'],x); B=ax(s['kappa_y'],s['vel_y'],s['Ly']/(ny-1),s['robin_y'],q)
 L=sparse.kron(sparse.eye(ny),A,format='csc')+sparse.kron(B,sparse.eye(nx),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
 return np.vstack([splu(L+mu*I).solve(src)[sensor] for mu in s['mu_shifts']]).T

if __name__=='__main__':
 kinds=['open','reflect','pin','periodic-local','periodic-mid','periodic-upwind']; candidates=list(product(kinds,repeat=2)); errors=np.array([rms(pred(*p)-y,1) for p in candidates]); order=np.argsort(np.max(errors,1));
 for j in order[:10]:
  p=candidates[j]; print(p,errors[j], 'shift RMSE',rms((pred(*p)-y).reshape(3,3,-1),2).round(6).tolist())
 for p in [('open','open'),('reflect','reflect')]: print('archive baseline',p,rms(pred(*p)-y,1))
 for k in range(3):
  j=np.argmin(np.mean(errors[:,np.arange(3)!=k]**2,1)); print('LOPO',k,candidates[j],errors[j,k])
 print('sensor range', y.min(),y.max()); print('sweeps',np.quantile(np.abs(s['sensor_calibration_sweeps']['raw_rosette']),[0.5,.9,.95,.98,1])); print('pulse data by row bank',np.unique(s['row_bank_id'],return_counts=True))
