import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; Y=np.c_[d['y'],d['traj']]; n=len(x)
def a(q):return np.column_stack((np.ones(len(q)),q))
def poly(q):return np.c_[a(q),*[q[:,i]*q[:,j] for i,j in it.combinations_with_replacement(range(q.shape[1]),2)]]
p=s['panel_a']; u,v=p['u'],p['v']; ii=np.ones(len(u),bool)
for _ in range(5):
 A=np.linalg.lstsq(a(u[ii]),v[ii],rcond=None)[0]; r=np.linalg.norm(a(u)@A-v,axis=1); ii=r<4*np.median(r)
A[1:]=np.diag(np.diag(A[1:])); p=s['panel_b']; B=np.linalg.lstsq(a(p['u']),p['v'],rcond=None)[0]; p=s['panel_c']; u=p['u']
def fit_c(key, f0, warp, bounds):
 v=p[key]
 def f(p,q):return a(warp(p,q))
 def res(p):
  m=f(p,u);return (m@np.linalg.lstsq(m,v,rcond=None)[0]-v).ravel()
 fit=least_squares(res,f0,bounds=bounds,ftol=1e-10,gtol=1e-10,xtol=1e-10); C=np.linalg.lstsq(f(fit.x,u),v,rcond=None)[0]; print(key,fit.x,np.sqrt(np.mean(fit.fun**2)),flush=True)
 return lambda q: f(fit.x,q)@C, lambda q:warp(fit.x,q)
C1,W1=fit_c('v1',[2.25,.45,0,0],lambda p,q:np.tanh(p[0]*q+p[2])+p[1]*q+p[3]*(1-np.cos(2.5*q)),([.2,-1,-1,-1],[5,2,1,1]))
C2,W2=fit_c('v2',[1.3,.65,.02],lambda p,q:q+p[1]*(1-np.cos(p[0]*q))+p[2]*q*q,([.2,-2,-2],[5,2,2]))
print('varying parameters:'); C1f,W1f=fit_c('v1',[2.25]*4+[.45,0],lambda p,q:np.tanh(q*p[:4]+p[5])+p[4]*q,([.2]*4+[-1,-1],[5]*4+[2,1])); C2f,W2f=fit_c('v2',[2.5,.2,.1],lambda p,q:q+p[1]*(1-np.cos(p[0]*q))+p[2]*q*q,([1.4,-2,-2],[5,2,2]))
ops={'a':lambda q:(q-A[0])/np.diag(A[1:]),'A':lambda q:a(q)@A,'B':lambda q:a(q)@B,'b':lambda q:(q-B[0])@np.linalg.inv(B[1:]),'c':C1,'d':C2,'e':C1f,'f':C2f}
frames={'':x}
for l in range(1,4):
 for seq in it.permutations(ops,l):
  if ('a'in seq and 'A'in seq) or ('b'in seq and 'B'in seq) or sum(c in 'cdef' for c in seq)>1:continue
  key=''.join(seq); frames[key]=ops[seq[-1]](frames[key[:-1]])
rng=np.random.default_rng(22); folds=[np.arange(56,72),*np.array_split(rng.permutation(n),5)]
def score(K,lam):
 rr=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); err=np.abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);rr.append([err[:,0].mean(),err[:,1:].mean()])
 return np.array(rr)
results=[]
def assess(k,q,mode):
 q=(q-q.mean(0))/np.maximum(q.std(0),1e-6); P=poly(q); K=P@P.T; r=score(K,1e-5);results.append((k,mode,0,0,r));
 if q.shape[1]>4:return
 D=cdist(q,q,'sqeuclidean')
 for h in (1.7,2.6,3.5):
  for lam in (.0002,.002):
   r=score(K+np.exp(-D/(2*h*h)),lam);results.append((k,mode,h,lam,r))
for k,q in frames.items():assess(k,q,'single')
for base in ('','a','A','B','b','aB','ab','AB','Ab','Ba','ba'):
 q=frames[base]
 for label,g in [('cd',np.c_[C1(q),C2(q)]),('basec',np.c_[q,C1(q)]),('based',np.c_[q,C2(q)]),('basecd',np.c_[q,C1(q),C2(q)])]:assess(base,g,label)
for out in (0,1):
 print('OUTPUT',out)
 for row in sorted(results,key=lambda r:r[-1][:,out].mean())[:15]:print(row[:-1],np.round(row[-1][:,out],5),np.round(row[-1][:,out].mean(),5))
