import itertools as it,pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_01 import _predict
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=3,suppress=True,linewidth=140)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(u):return np.column_stack([np.ones(len(u)),u,*[u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(4),2)]])
def loo(a,b):
 p=np.linalg.pinv(a);e=b-a@(p@b);return np.mean(abs(e/(1-(a*p.T).sum(1))[:,None]),0)
a=s['panel_a']; u=a['u']; A=np.c_[np.ones(len(u)),u]; w=ls(A,a['v']);
for _ in range(10):
 r=np.linalg.norm(a['v']-A@w,axis=1); wt=np.sqrt(np.minimum(1,1.5*np.median(r)/np.maximum(r,1e-15)));w=ls(A*wt[:,None],a['v']*wt[:,None])
r=np.linalg.norm(a['v']-A@w,axis=1);keep=np.argsort(r)[:-3];w=ls(A[keep],a['v'][keep]); print('panel A',w,'row residuals',np.sort(np.linalg.norm(a['v']-A@w,axis=1))[-8:]);
x=s['train']['x']; z=np.linalg.solve(w[1:].T,(x-w[0]).T).T;c=s['panel_c']; u=c['u']; p=poly(u); v1=c['v1'];v2=c['v2'];print('v1 coefficients',ls(p,v1).T);q=ls(p,v2);print('v2 coefficients',q.T);print('v2 extrema',v2.min(0),v2.max(0));
for j in range(4):
 m=np.zeros((4,4));
 for h,(k,l) in enumerate(it.combinations_with_replacement(range(4),2)):m[k,l]=m[l,k]=q[5+h,j]/(2 if k!=l else 1)
 print('quad eig',np.linalg.eigvalsh(m))
f1=poly(z)@ls(p,v1);f2=poly(z)@q
T=s['train']; folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(np.random.default_rng(123).permutation(72),4)]
def ev(F):
 return np.array([[np.mean(abs(_predict(F,T[t],a,b,l)-T[t][b])) for t,l in [('y',.003),('traj',.0001)]] for a,b in folds])
for n,f in [('z',z),('quadratic',poly(z)[:,1:]),('v1',f1),('v1 z',np.c_[f1,z]),('v1 v2',np.c_[f1,f2]),('v1 v2 z',np.c_[f1,f2,z])]:
 v=ev(f); print(n,'official',v[0],'worst',v.max(0))
