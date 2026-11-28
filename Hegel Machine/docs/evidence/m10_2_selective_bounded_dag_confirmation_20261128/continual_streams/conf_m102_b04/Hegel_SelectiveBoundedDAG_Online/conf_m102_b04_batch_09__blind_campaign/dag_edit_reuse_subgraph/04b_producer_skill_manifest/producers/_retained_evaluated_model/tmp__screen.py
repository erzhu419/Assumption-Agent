import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; r=run_node({'__input__':s}); x=np.cbrt(r['cubic_train']); raw=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(y); Y=np.c_[y,t]
def aff(q):return np.c_[np.ones(len(q)),q]
def fit(u,v,basis):
 D=basis(u); return np.linalg.lstsq(D,v,rcond=None)[0]
def compare(panel,key,forms):
 u=s[panel]['u'];v=s[panel][key]
 out=[]
 for nm,fn,lo,hi in forms:
  def err(p):
   D=fn(u,p); return np.mean((D@np.linalg.lstsq(D,v,rcond=None)[0]-v)**2)
  o=minimize_scalar(err,bounds=(lo,hi),method='bounded');print(panel,key,nm,round(o.x,6),round(o.fun**.5,7));out.append((o.fun,nm,fn,o.x))
 _,nm,fn,p=min(out);return lambda q:fn(q,p)@fit(u,v,lambda z:fn(z,p))
b=s['panel_b']; bm=lambda q:aff(q)@fit(b['u'],b['v'],aff)
vm=compare('panel_c','v1',[(k,lambda q,p,f=f:aff(f(p*q)),.05,5) for k,f in [('atan',np.arctan),('tanh',np.tanh),('sin',np.sin),('asinh',np.arcsinh)]]+ [('soft',lambda q,p:aff(q/np.sqrt(1+p*q*q)),.01,10)])
wm=compare('panel_c','v2',[(k,fn,.001,5) for k,fn in [('rat',lambda q,p:np.c_[aff(q),q*q/(1+p*q*q)]),('exp',lambda q,p:np.c_[aff(q),np.expm1(-p*q*q)]),('cos',lambda q,p:np.c_[aff(q),np.cos(p*q)]),('sqrt',lambda q,p:np.c_[aff(q),np.sqrt(1+p*q*q)])]])
coords={'x':x,'raw':raw,'b':bm(x),'br':bm(raw),'v':vm(x),'vr':vm(raw),'vb':vm(bm(x)),'w':wm(x),'wr':wm(raw),'wb':wm(bm(x)),'wbr':wm(bm(raw))};
def poly(q,deg):return np.column_stack([np.prod(q[:,j],axis=1) for d in range(1,deg+1) for j in combinations_with_replacement(range(q.shape[1]),d)])
def errors(K,lam):
 A=K+1.;A.flat[::n+1]+=lam; P=cho_solve(cho_factor(A),np.eye(n)); e=abs((P@Y)/P.diagonal()[:,None]); return np.r_[e[:,0].mean(),e[:,1:].mean()]
if __name__=='__main__':
 sets=dict(coords);sets.update({k:np.c_[coords[k[0]],coords[k[1:]]] for k in ['vw','vwb','xv','xw','xwb']});sets['all']=np.hstack(list(coords.values()))
 out=[]
 for name,q in sets.items():
  q=q/q.std(0); ds=cdist(q,q,'sqeuclidean')/q.shape[1]; dot=q@q.T/q.shape[1]; ks={}
  for d in [1,2,3,4,5,6]:ks[f'p{d}']=(1+dot)**d
  for a in [.03,.08,.18,.35,.65,1.2,2.5]:
   ks[f'rbf{a}']=np.exp(-a*ds)
   parts=np.exp(-a*(q[:,None,:]-q[None,:,:])**2);ks[f'add{a}']=parts.mean(2);ks[f'pair{a}']=(parts.sum(2)**2-(parts**2).sum(2))/(q.shape[1]*(q.shape[1]-1))
  for k,K in ks.items():
   for lam in [1e-7,1e-5,1e-4,.001,.01,.05,.2]:out.append((*errors(K,lam),name,k,lam))
 for axis in [0,1]:
  print('best',axis);print(*sorted(out,key=lambda r:r[axis])[:18],sep='\n')
