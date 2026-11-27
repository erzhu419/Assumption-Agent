import contextlib, io, runpy, itertools
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path('/tmp/refine_chamber.py')
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
mem=(hold[:,None]-z)*theta[:2]; f=-np.expm1(-t); inferred=np.median((y-mem@decay.T)/f,axis=1); base=theta[2]+xdiag@theta[3:]; target=inferred-base
lab=di['lab_id']; st=di['stratum']; missing=np.isnan(di['coupon_features_raw']); print('i lab stratum missing x0..x3 hold z0 z1 base inferred delta'); print(np.round(np.c_[np.arange(len(r)),lab,st,missing.sum(1),xdiag,hold,z,base,inferred,target],4)); print('diagnostic extra keys',list(di))
terms={}
def add(name,v):
 if np.ptp(np.asarray(v, dtype=float))>1e-8: terms[name]=np.asarray(v,dtype=float)
for k in range(4):
 a=xdiag[:,k]; add(f'x{k}',a); add(f'x{k}^2',a*a); add(f'abs(x{k})',abs(a)); add(f'pos(x{k})',np.maximum(a,0)); add(f'neg(x{k})',np.maximum(-a,0)); add(f'missing{k}',missing[:,k])
for k,j in itertools.combinations(range(4),2):
 a,b=xdiag[:,k],xdiag[:,j]; add(f'x{k}*x{j}',a*b); add(f'(x{k}+x{j})^2',(a+b)**2); add(f'(x{k}-x{j})^2',(a-b)**2); add(f'abs(x{k}-x{j})',abs(a-b)); add(f'abs(x{k}+x{j})',abs(a+b))
for k,j,l in itertools.combinations(range(4),3): add(f'x{k}*x{j}*x{l}',xdiag[:,k]*xdiag[:,j]*xdiag[:,l])
for k in np.unique(st): add(f'stratum{k}',st==k)
for name,v in [('hold',hold),('z0',z[:,0]),('z1',z[:,1]),('base',base),('base^2',base**2),('base^3',base**3),('mem0',mem[:,0]),('mem1',mem[:,1])]: add(name,v)
for k in range(past.shape[1]): add(f'past{k}',past[:,k])
for i in range(4):
 for j in range(4): add(f'x{i}/(1+x{j}^2)',xdiag[:,i]/(1+xdiag[:,j]**2))
for k,v in list(terms.items()):
 if k not in ('base','hold'): add('drift*'+k,base*v)

def score(v,a=target):
 errs=[]
 for j in np.unique(lab):
  test=lab==j; q=~test; mat=np.c_[np.ones(len(v)),v]; b=np.linalg.lstsq(mat[q],a[q],rcond=None)[0]; errs.extend((i,e) for i,e in zip(np.flatnonzero(test),abs(a[test]-mat[test]@b)))
 return np.mean([v for _,v in errs]),max(np.mean([e for i,e in errs if lab[i]==j]) for j in np.unique(lab))
print('baseline mean/worst diagnostic drift',np.mean(abs(target)),max(np.mean(abs(target[lab==j])) for j in np.unique(lab)))
print('single-term screening (not nested performance estimates)'); print(*sorted((score(v),k) for k,v in terms.items())[:22],sep='\n')
print('rate minus curve-implied amplitude rms',np.sqrt(np.mean((r-(mem/taus).sum(1)-inferred)**2)))
