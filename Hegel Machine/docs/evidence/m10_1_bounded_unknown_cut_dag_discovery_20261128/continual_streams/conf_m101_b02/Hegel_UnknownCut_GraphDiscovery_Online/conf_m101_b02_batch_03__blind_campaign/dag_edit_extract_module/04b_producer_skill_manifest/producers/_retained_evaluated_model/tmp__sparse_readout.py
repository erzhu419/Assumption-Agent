exec(open('/tmp/dag_probe.py').read().split("if '--setup'")[0])
from sklearn.linear_model import ElasticNet
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
H=-np.expm1(-t[:,None]*np.array([.4167,1.8204])); Y=np.c_[y,T@np.linalg.pinv(H.T)]
print('ranges',[(w.min(0),w.max(0)) for w in (x,z,b,c,d)]); print('linear',np.linalg.lstsq(L(z),Y,rcond=None)[0],flush=True)
F={}
def add(name,v): F[name]=v
for nm,w in [('z',z),('x',x),('b',b),('f',f),('c',c),('d',d)]:
 for js in combinations_with_replacement(range(4),3):add((nm,'cubic',js),np.prod(w[:,js],axis=1))
 for j,v in enumerate(w.T):
  for k in [.5,1.,1.5,2.,3.]:
   for name,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('soft',lambda q:q/(1+abs(q))),('rat',lambda q:q/(1+q*q))]:add((nm,name,j,k),fun(k*v))
V=[(nm,j,v) for nm,q in [('z',z),('x',x),('b',b),('f',f),('c',c),('d',d)] for j,v in enumerate(q.T)]
for i,(nm,j,v) in enumerate(V):
 for nn,k,w in V[i:]:
  add((nm,nn,j,k,'prod'),v*w)
  for h in [1.,2.]:
   for name,term in [('sin+',np.sin(h*(v+w))),('sin-',np.sin(h*(v-w))),('cos+',np.cos(h*(v+w))),('cos-',np.cos(h*(v-w))),('th+',np.tanh(h*(v+w))),('th-',np.tanh(h*(v-w))),('div',v/(1+h*w*w)),('rdiv',w/(1+h*v*v)),('multh',np.tanh(h*v*w))]:add((nm,nn,j,k,name,h),term)
  for name,term in [('cub21',v*v*w),('cub12',v*w*w),('sat1',np.tanh(v)*w),('sat2',v*np.tanh(w))]:add((nm,nn,j,k,name),term)
keys=list(F); X=np.array(list(F.values())).T; rng=np.random.default_rng(25)
# Reproduce the evaluator split as well as testing separate resampled folds.
folds=[(np.arange(52),np.arange(52,72)),(np.arange(54),np.arange(54,72)),(np.arange(56),np.arange(56,72)),(np.arange(48),np.arange(48,72))]+[(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(rng.permutation(n),4)]
errors={}
for ix,(a,v) in enumerate(folds):
 P=L(z); R=X-P@np.linalg.lstsq(P[a],X[a],rcond=None)[0]; keep=np.std(R[a],axis=0)>1e-7; R=R[:,keep]; R/=R[a].std(0); names=np.array(keys,dtype=object)[keep]; base=P@np.linalg.lstsq(P[a],Y[a],rcond=None)[0]; Q=Y-base
 options={}
 for ratio in [.1,.4,.8,1.]:
  for al in [.001,.005,.015,.04]:
   ps=[]
   for j in range(3):
    sc=Q[a,j].std(); m=ElasticNet(alpha=al,l1_ratio=ratio,max_iter=1500,tol=.001,fit_intercept=False).fit(R[a],Q[a,j]/sc); ps.append(m.predict(R[v])*sc)
    if ix==0 and ratio==1 and al==.015:
     ids=np.argsort(abs(m.coef_))[-8:][::-1];print('support',j,[(names[k],round(m.coef_[k],3)) for k in ids])
   options[('elastic',ratio,al)]=base[v]+np.array(ps).T
 for cut in [3,6,10,15]:
  pred=[]
  for yy in Q.T:
   ids=[]; res=yy[a]; M=R[a].copy()
   for _ in range(cut):
    ns=np.linalg.norm(M,axis=0); j=np.argmax(np.abs(res@M)/np.maximum(ns,1e-8)); ids.append(j); U=np.linalg.qr(R[a][:,ids])[0]; res=yy[a]-U@(U.T@yy[a]); M=R[a]-U@(U.T@R[a]);
   pred.append(R[v][:,ids]@np.linalg.lstsq(R[a][:,ids],yy[a],rcond=None)[0]);
   if ix==0 and cut==10:print('omp',[(names[j]) for j in ids])
  options[('omp',cut)]=base[v]+np.array(pred).T
 for config,p in options.items():
  e=np.array([abs(p[:,0]-y[v]).mean(),abs(p[:,1:]@H.T-T[v]).mean()]);errors.setdefault(config,[]).append(e)
for k in range(2):
 for label,score in [('suffix',lambda e:e[0][k]),('worst',lambda e:max(q[k] for q in e))]:print(label,k,[(m,np.round(e,4).tolist()) for m,e in sorted(errors.items(),key=lambda p:score(p[1]))[:3]],flush=True)
