exec(open('/tmp/sink_probe.py').read().split('for label,A in')[0])
from itertools import combinations_with_replacement as cwr
from itertools import combinations
ids=np.arange(N); y=Y[:,0]
vs=[folds[0][1]]+[f for seed in (7,41,113,307) for f in np.array_split(np.random.default_rng(seed).permutation(ids),4)]+list(np.array_split(ids,4))+[np.argsort(v)[::d][:12] for v in np.c_[x,z].T for d in (1,-1)]
fs=[(np.setdiff1d(ids,v),v) for v in vs]
def score(D):
 a=np.array([np.mean(abs(D[j]@np.linalg.lstsq(D[i],y[i],rcond=None)[0]-y[j])) for i,j in fs]);return np.round([a[0],a[1:21].max(),a[21:].max(),a.mean()],5)
A=np.c_[x,z]; base=np.c_[np.ones(N),z]; lib={}
for k in (2,3):
 for ix in cwr(range(8),k):
  lib[''.join('abcdefgh'[i] for i in ix)]=np.prod(A[:,ix],axis=1)
for i,u in enumerate(A.T):
 lib['lin'+str(i)]=u
 for j in range(i+1,8):
  v=A[:,j]
  for s in (-1,1):
   w=u+s*v
   for nm,val in [('sin',np.sin(w)),('tanh',np.tanh(w)),('rat', w/(1+w*w)),('cos',np.cos(w))]:lib[f'{nm}({i},{j},{s})']=val
names=list(lib); M=np.column_stack(list(lib.values()));
# Beam search uses the official training prefix only; test errors are not used to rank the beam.
tr=fs[0][0]
for B in (base,np.c_[base,x]):
 beam=[()];print('base',score(B))
 for size in range(1,9):
  best={}
  for ix in beam:
   D=np.c_[B,M[:,ix]]; d=D[tr]; q=np.linalg.qr(d)[0]; r=y[tr]-q@(q.T@y[tr]); rem=M[tr]-q@(q.T@M[tr]); norm=np.linalg.norm(rem,axis=0); gain=(r@rem)**2/np.maximum(norm**2,1e-30); gain[norm<1e-8]=0
   for k in np.argsort(gain)[-18:]:
    if k not in ix:
     key=tuple(sorted((*ix,k))); best[key]=r@r-gain[k]
  beam=sorted(best,key=best.get)[:72]; ix=beam[0]; D=np.c_[B,M[:,ix]]; print(size,round(best[ix],5),[names[k] for k in ix],score(D))
