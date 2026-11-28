exec(open('/tmp/wall_probe.py').read().split('folds=')[0])
from scipy.sparse import csc_matrix
allix=np.arange(len(y)); folds=[allix[allix//ns==k] for k in range(len(s['mu_shifts']))]
ro, se, pe=[axes((mode,mode)) for mode in ('robin','sealed','periodic')]

def mixed(a,b):
 out=[]
 for d, v in enumerate((a,b)):
  M=se[d].toarray(); choices=(se[d].toarray(),pe[d].toarray(),ro[d].toarray())
  M[0]=choices[v[0]][0]; M[-1]=choices[v[1]][-1]; out.append(csc_matrix(M))
 return out

def score(S):
 e=np.empty_like(y)
 for te in folds:
  tr=np.setdiff1d(allix,te); w,g=fit(S,tr,1e-8); e[te]=y[te]-S[te]@w-z[te]@g
 w,g=fit(S,allix,1e-8); rm=np.sqrt(np.mean((y-S@w-z@g)**2)); rmse=[np.sqrt(np.mean(e[te]**2)) for te in folds]; return np.sqrt(np.mean(e**2)),rm,rmse
out=[]
for v in itertools.product(range(3),repeat=4):
 S,_=response._responses(s,mixed(v[:2],v[2:])); out.append((score(S),v))
for sc,v in sorted(out)[:9]: print(v,sc)
print('nuisance',z.shape, 'ridge',lam,'sensor locations at walls',[(sensor[:,i]==0).sum()+(sensor[:,i]==s['n'+d]-1).sum() for i,d in enumerate('xy')])
