import numpy as np,pickle
from itertools import combinations_with_replacement,combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=np.load('/tmp/probe.npz')['u']; n=len(z); yy=s['train']['y']; tr=s['train']['traj']; _,_,vt=np.linalg.svd(tr,full_matrices=False); Y=np.column_stack((yy,tr@vt[:2].T)); P=np.load('/tmp/struct6.npz'); d={}
def add(p,x):
 for j,c in enumerate(x.T):d[p+str(j)]=c
add('z',z)
for k in (2,3,4):
 for p in combinations_with_replacement(range(4),k):d['z'+''.join(map(str,p))]=z[:,p].prod(1)
for f,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda z:z/(1+z*z))]:
 for k in (1,2,3):add(f+str(k)+'z',fun(k*z))
for p in P.files:
 v=P[p];add(p,v);add(p+'sin',np.sin(v));add(p+'tanh',np.tanh(v));add(p+'square',v*v)
 for j in range(4):
  for k in range(4):d[f'{p}{j}*z{k}']=v[:,j]*z[:,k]
 for j,k in combinations(range(4),2):d[f'{p}{j}*{k}']=v[:,j]*v[:,k]
for p,q in combinations(P.files,2):
 for j in range(4):
  for k in range(4):d[f'{p}{j}*{q}{k}']=P[p][:,j]*P[q][:,k]
D=np.column_stack(list(d.values())); names=np.array(list(d));
def sparse(idx,y,m):
 a=np.column_stack((np.ones(n),z)); rem=list(range(D.shape[1])); chosen=[]
 for _ in range(m):
  q=np.linalg.qr(a[idx],mode='reduced')[0]; r=y[idx]-q@(q.T@y[idx]); w=D[np.ix_(idx,rem)]; w=w-q@(q.T@w); j=rem.pop(np.argmax(abs(r@w)/np.maximum(1e-10,np.linalg.norm(w,axis=0)))); chosen.append(j); a=np.column_stack((a,D[:,j]))
 return a, names[chosen]
for j,y in enumerate(Y.T):
 for m in (3,6,10,16):
  a,names2=sparse(np.arange(52),y,m); b=np.linalg.lstsq(a[:52],y[:52],rcond=None)[0]; e=a@b-y; print('sparse',j,m,round(abs(e[:52]).mean(),4),round(abs(e[52:]).mean(),4),list(zip(names2,b[5:].round(3))))
# Profile common nonlinearities with independently identified channel gains.
for panel,key,kind,k0 in [('b','v','sin',2.8),('c','v1','tanh',2.28),('c','v2','cos',1.28)]:
 p=s['panel_'+panel]; x,y=p['u'],p[key]; fun=getattr(np,kind)
 for extra in (False, True):
  def h(x,k):return np.column_stack((np.ones(len(x)),x,fun(x*k),*([x*x] if extra else [])))
  def fit(k,ret=False):
   a=h(x,k); b=np.linalg.lstsq(a,y,rcond=None)[0]; e=a@b-y
   return (b,e) if ret else e.ravel()
  o=least_squares(fit,np.full(4,k0),bounds=(.2,5.5)); b,e=fit(o.x,True); print(panel,key,extra,o.x.round(4),np.mean(abs(e)),np.linalg.svd(e,compute_uv=False).round(5));
