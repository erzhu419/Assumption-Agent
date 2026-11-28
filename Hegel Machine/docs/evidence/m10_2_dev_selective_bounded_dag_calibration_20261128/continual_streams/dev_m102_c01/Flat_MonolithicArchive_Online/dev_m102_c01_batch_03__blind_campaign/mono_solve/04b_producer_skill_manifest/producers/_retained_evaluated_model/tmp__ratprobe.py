import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
idx=np.arange(72); splits=[(idx[:52],idx[52:])]+[(np.setdiff1d(idx,k),k) for k in np.array_split(np.random.default_rng(74).permutation(72),4)]
for col in range(3):
 yy=target[:,col]; rows=[]
 for name,v in coords.items():
  for m in (2,3,4,5):
   for dd in (0,1,2):
    F=PolynomialFeatures(m).fit_transform(v); Q=PolynomialFeatures(max(dd,1)).fit_transform(v)[:,1:] if dd else v[:,:0]; P=np.c_[F,-yy[:,None]*Q];
    for r in (.0001,.003,.03,.3):
     sc=np.maximum(np.linalg.norm(P,axis=0),1e-6); R=P/sc; K=R@R.T; es=[]; ds=[]
     for i,j in splits:
      w=(R[i].T@solve(K[np.ix_(i,i)]+r*np.eye(len(i)),yy[i],assume_a='pos'))/sc; den=1+Q[j]@w[F.shape[1]:]; yh=(F[j]@w[:F.shape[1]])/np.maximum(.2,den); es.append(np.mean(abs(yh-yy[j]))); ds.append(min(den))
     rows.append((np.mean(es[1:]),es[0],max(es),name,m,dd,r,min(ds)))
 print('rat',col,sorted(rows)[:12]); print('gate',sorted(rows,key=lambda v:v[1])[:6])
print('archived direct',end=' ')
from arc import solve_batch
for i,j in splits:
 ss=dict(s,train={k:v[i] if k!='summary' else v for k,v in tr.items()},test={'x':x[j]}); out=solve_batch(ss); print([np.mean(abs(out[k]-a[j])) for k,a in [('y_hat',y),('state_pred',t)]],end=' ')
print()
