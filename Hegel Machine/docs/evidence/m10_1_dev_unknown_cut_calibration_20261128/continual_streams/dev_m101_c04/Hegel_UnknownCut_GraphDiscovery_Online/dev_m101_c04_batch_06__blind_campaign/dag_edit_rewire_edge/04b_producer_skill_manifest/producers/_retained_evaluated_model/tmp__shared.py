from dag_probe import *
from itertools import combinations_with_replacement,product
from sklearn.linear_model import Lasso
from scipy.optimize import least_squares
F=basis(u,2); N=52; blocks=[np.arange(52,72),np.arange(0,20),np.arange(20,40),np.arange(40,60)]; folds=[(np.setdiff1d(np.arange(len(u)),v),v) for v in blocks]; X=np.c_[u,z]; L=basis(u,1)
def val(A,alpha=0.):
 e=[]
 for t,v in folds:
  w=np.linalg.lstsq(A[t],Y[t],rcond=1e-9)[0] if not alpha else np.linalg.solve(A[t].T@A[t]+alpha*np.eye(A.shape[1]),A[t].T@Y[t]); e.append(np.mean(abs(A[v]@w-Y[v])))
 return np.round(e,5)
print('shared amplitude fit',val(np.c_[L,R[:,-1]]),np.linalg.lstsq(np.c_[L,R[:,-1]][:N],Y[:N],rcond=None)[0]); print('quad vs traj quadratic',np.linalg.lstsq(F[:N],np.c_[Y,R[:,-1]][:N],rcond=None)[0][5:].T.round(3))
D={}
def add(name,g): D[name]=g
for i,j,k in combinations_with_replacement(range(8),3):
 for a in [0,.08,1/6,.35,.7]:
  base=X[:,i]*X[:,j]*X[:,k]; add(f'{i}{j}{k}/{a}:{k}',base/(1+a*X[:,k]**2))
  if a:
   for m in {i,j}: add(f'{i}{j}{k}/{a}:{m}',base/(1+a*X[:,m]**2))
for i,j in product(range(8),repeat=2):
 for a in [.15,.4,.8,1.5]:
  for sign in [-1,1]:
   w=X[:,i]+sign*X[:,j]
   for k in [i,j]:
    add(f'{k}*rat({i},{j},{sign},{a})',X[:,k]*w**2/(1+a*w**2)); add(f'tanh({i},{j},{sign},{a})',np.tanh(a*w))
  add(f'{i}*sin({j},{a})',X[:,i]*np.sin(a*X[:,j])); add(f'{i}*cos({j},{a})',X[:,i]*np.cos(a*X[:,j]))
for i,j,k in product(range(4),repeat=3):
 for a in [.1667,.4,.8]:
  add(f'odd/{i}{j}{k}/{a}',u[:,i]*z[:,j]/(1+a*X[:,k]**2)); add(f'odd/z{i}{j}{k}/{a}',u[:,i]*z[:,j]/(1+a*z[:,k]**2))
G=np.column_stack(list(D.values())); names=list(D)
for lab,A in [('quad',F),('shared',np.c_[L,R[:,-1]])]:
 for shortlist in [False,True]:
  J=np.array([j for j,k in enumerate(names) if not shortlist or ('/' in k and ': ' not in k and ('/0:' in k or '/0.16666666666666666:' in k))]); H=G[:,J]; out=[]; selected=[]
  for t,v in folds:
   h=H-A@np.linalg.lstsq(A[t],H[t],rcond=None)[0]; r=Y-A@np.linalg.lstsq(A[t],Y[t],rcond=None)[0]; sd=np.maximum(1e-10,np.linalg.norm(h[t],axis=0)); h/=sd; e=[]; ids=[]
   for j in range(12):
    coef=np.linalg.lstsq(h[t][:,ids],r[t],rcond=None)[0]; rr=r[t]-h[t][:,ids]@coef; best=np.argmax(abs(h[t].T@rr)); ids.append(best); coef=np.linalg.lstsq(h[t][:,ids],r[t],rcond=None)[0]; e.append(np.mean(abs(h[v][:,ids]@coef-r[v])))
   out.append(e); selected.append([names[J[j]] for j in ids]);
  print(lab,shortlist,'omp',np.array(out).T.round(4).tolist(),'terms',selected[0]);
  out=[]
  for t,v in folds:
   h=H-A@np.linalg.lstsq(A[t],H[t],rcond=None)[0]; r=Y-A@np.linalg.lstsq(A[t],Y[t],rcond=None)[0]; sd=np.maximum(1e-10,np.sqrt(np.mean(h[t]**2,axis=0))); h/=sd; e=[]
   for alpha in [.001,.002,.004,.008,.015]:
    reg=Lasso(alpha=alpha,max_iter=4000,tol=1e-4).fit(h[t],r[t]); e.append(np.mean(abs(reg.predict(h[v])-r[v])));
    if t[0]==0 and v[0]==52 and alpha==.004: print('lasso terms',[(names[J[j]],round(reg.coef_[j],3)) for j in np.argsort(abs(reg.coef_))[-8:]])
   out.append(e)
  print(lab,shortlist,'lasso',np.array(out).T.round(4).tolist())
