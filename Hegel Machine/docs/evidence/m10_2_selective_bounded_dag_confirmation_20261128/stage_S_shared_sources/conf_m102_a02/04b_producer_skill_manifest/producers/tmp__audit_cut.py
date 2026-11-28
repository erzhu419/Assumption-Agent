import itertools,pickle
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; h=s['train']['traj']; t=s['eval_times']; q=list(itertools.combinations_with_replacement(range(4),2)); a=np.c_[np.ones(len(x)),x]; p=np.column_stack([a]+[x[:,i]*x[:,j] for i,j in q]); ix=np.arange(len(x)); rng=np.random.default_rng(555)
print('Study fields:', sorted(s), 'training fields:', sorted(s['train'])); print('Public-test coordinate range:', np.min(s['test']['x'],0), np.max(s['test']['x'],0)); print('Train range:', x.min(0),x.max(0))
print('Pairwise difference square span check:',max(np.max(abs((x[:,i]-2*x[:,j])**2-p@np.linalg.lstsq(p,(x[:,i]-2*x[:,j])**2,rcond=None)[0])) for i,j in itertools.combinations(range(4),2)))
splits=[ix[52:],*np.array_split(rng.permutation(ix),4),*[np.argsort(abs(x[:,k]-x[:,k].mean()))[-15:] for k in range(4)],np.argsort(np.linalg.norm(x,axis=1))[-15:]]
for fold,j in enumerate(splits):
 i=np.setdiff1d(ix,j); w=np.linalg.lstsq(p[i],y[i],rcond=None)[0]; base=p[j]@w
 # Positive denominator, testing coordinate, pairwise sum/difference and arbitrary linear directions.
 def residual(b):
  d=1+(x@b)**2; f=p/d[:,None]; c=np.linalg.lstsq(f[i],y[i],rcond=None)[0]; return (f@c-y)[i],(f@c)[j]
 seeds=list(np.eye(4))+[np.eye(4)[u]+sign*np.eye(4)[v] for u,v in itertools.combinations(range(4),2) for sign in [-1,1]]
 fit=min((least_squares(lambda b:residual(b)[0], .06*seed, bounds=(-2,2),max_nfev=40) for seed in seeds),key=lambda r:r.cost)
 W=np.linalg.lstsq(a[i],h[i],rcond=None)[0]; pred=a[j]@W; V=np.linalg.svd(a[i]@W,full_matrices=False)[2][:2]; reduced=pred@V.T@V
 # Candidate two-mode constant-coefficient step response with zero initial response.
 def expfit(log_rates):
  B=-np.expm1(-np.exp(log_rates)[:,None]*t); C=np.linalg.lstsq(B.T,W.T,rcond=None)[0].T; return ((a[i]@C)@B-h[i]).ravel(),a[j]@C@B
 exp=min((least_squares(lambda r:expfit(r)[0],np.log(seed),bounds=(-5,4)) for seed in [(0.3,2),(0.1,1),(1,4)]),key=lambda r:r.cost)
 e0=np.abs(base-y[j]).mean(); e1=np.abs(residual(fit.x)[1]-y[j]).mean(); e2=np.abs(reduced-h[j]).mean(0).max(); e3=np.abs(expfit(exp.x)[1]-h[j]).mean(0).max()
 print(f'fold={fold} endpoint quadratic={e0:.6f} rational={e1:.6f}; worst horizon rank2={e2:.6f} two_rate={e3:.6f}; rates={np.exp(exp.x)}')
