from chamber_probe import *
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
past, hold, drift = inp_d
A,G=m._design(tau,inp_d,dt,t)
obs=np.c_[yd-A@th,rd-G@th]
def bt(lam):
 return np.vstack((np.column_stack((-np.expm1(-t[:,None]/lam),t)),np.r_[1/lam,1.]))
print('residual singular values',np.linalg.svd(obs,compute_uv=False).round(5))
B=bt(tau); amps=np.linalg.lstsq(B,obs.T,rcond=None)[0].T
print('fixed tau residual RMSE by time/rate',np.sqrt(np.mean((obs-amps@B.T)**2,axis=0)).round(5))
print('per-coupon',np.c_[hold,past,amps].round(4))
Y=np.c_[yd-(drift@th[2:])[:,None]*t,rd-drift@th[2:]]
def fixed_drift(logtau):
 b=bt(np.sort(np.exp(logtau)))[:,:2]; return (np.linalg.lstsq(b,Y.T,rcond=None)[0].T@b.T-Y).ravel()
print('known drift fixed taus',score(fixed_drift(np.log(tau))))
fit=least_squares(fixed_drift,np.log(tau),bounds=(np.log([.05,.05]),np.log([25,25])))
print('known drift optimized taus',np.sort(np.exp(fit.x)),score(fit.fun))
mem = hold[:,None] - G[:,:2]*tau
D=np.c_[hold,past,mem,drift[:,1:]]
names=['hold',*[f'p{i}' for i in range(past.shape[1])],'z0','z1',*[f'x{i}' for i in range(drift.shape[1]-1)]]
terms=[j for k in (1,2,3) for j in combinations_with_replacement(range(len(names)),k)]
V=np.column_stack([np.prod(D[:,j],axis=1) for j in terms]); V-=V.mean(axis=0); V/=np.linalg.norm(V,axis=0)
for j,a in enumerate(amps.T):
 c=V.T@(a-a.mean())/np.linalg.norm(a-a.mean()); best=np.argsort(-abs(c))[:12]; print('fixed-tau amplitude',j,'mean/std',a.mean(),a.std(),'top',[(tuple(names[i] for i in terms[k]),round(c[k],4)) for k in best]); print('linear load regression',np.linalg.lstsq(np.c_[np.ones(len(D)),D[:,:7]],a,rcond=None)[0].round(5),'residual SD',np.linalg.lstsq(np.c_[np.ones(len(D)),D[:,:7]],a,rcond=None)[1]**.5/len(D)**.5)
from pathlib import Path
print('archived mechanism hints:')
for f in Path('/tmp/hegel_continual_archive').glob('*solver.py'):
 print(f.name)
 for line in f.read_text().splitlines():
  if any(w in line.lower() for w in ('coupl','chamber','interact','drift','memory','withdraw')): print(line)
