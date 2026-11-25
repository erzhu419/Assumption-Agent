import sys, time, itertools, pickle
import numpy as np
from scipy.optimize import lsq_linear
sys.path.insert(0, '/holdout_task')
from certified_l0_solver import solve_certified_box_l0

def inst(X,y,r,b,l): return dict(X=X,y=y,ridge=r,box=b,l0_penalty=l)
def oracle(a):
 X,y,r,b,l=[a[k] for k in ('X','y','ridge','box','l0_penalty')]; p=X.shape[1]
 A=np.vstack((X,np.sqrt(r)*np.eye(p))); d=np.r_[y,np.zeros(p)]
 lower=upper=np.inf
 for mask in itertools.product((False,True),repeat=p):
  lim=np.where(mask,b,min(b,1e-9))
  z=np.clip(lsq_linear(A,d,bounds=(-lim,lim),tol=1e-13).x,-lim,lim)
  res=A@z-d; g=A.T@res; loss=.5*(res@res)
  lower=min(lower,loss-g@z-abs(g)@lim+l*sum(mask)); upper=min(upper,loss+l*np.count_nonzero(abs(z)>1e-9))
 return lower,upper

def check(name,a,ref=None):
 t=time.perf_counter()
 try:
  z=solve_certified_box_l0(a); w=np.asarray(z['weights']); d=a['X']@w-a['y']; value=.5*(d@d+a['ridge']*(w@w))+a['l0_penalty']*np.count_nonzero(abs(w)>1e-9); lb=z['certified_lower_bound']; gap=value-lb
  assert w.shape==(a['X'].shape[1],) and np.all(abs(w)<=a['box']) and abs(value-z['objective'])<=1e-11*(1+value)
  assert -1e-11<=gap<=1e-5 and 0<=z['nodes_explored']<=600
  if ref is not None: assert lb<=ref[1]+1e-10 and value<=ref[0]+1.001e-5, (lb,value,ref)
  return z['nodes_explored'],gap,time.perf_counter()-t
 except Exception as e:
  print('FAIL',name,repr(e),round(time.perf_counter()-t,3),flush=True); return None

def main():
 for i,a in enumerate(pickle.load(open('/holdout_task/public_box_l0_instances.pkl','rb'))): print('public',i,check(i,a))
 rng=np.random.default_rng(318)
 small=[inst(rng.normal(size=(6,6)),rng.normal(size=6),.01,.15,.12),inst(np.eye(4)*1e5,np.array([1e-4,2e-4,-7e-4,0]),.1,1e-8,1e-8),inst(np.eye(4)*1e5,np.ones(4),.1,5e-10,.1)]
 for i,a in enumerate(small):
  ref=oracle(a); print('oracle',i,ref,check(i,a,ref))
 for kind in ('dense','under','dup','orth','block','rank1'):
  rows=[]
  for k in range(10 if kind in ('dense','under','dup') else 3):
   n={'dense':28,'under':5,'dup':20}.get(kind,15); X=rng.normal(size=(n,15)); y=rng.normal(size=n)
   if kind=='dup': X[:,7:14]=X[:,:7]+.005*X[:,7:14]; y=X@rng.normal(size=15)*.35+y*.15
   if kind=='orth': X=np.eye(15); y=np.ones(15)*np.sqrt(1.01*.8)
   if kind=='block': X=np.kron(np.eye(5),np.ones((3,3))); y=np.ones(15)*.7
   if kind=='rank1': X=np.ones((15,15)); y=np.ones(15)*(1.2+k*.4)
   a=inst(X,y,(.01,.1,.35)[k%3],(.35,.8,1.5)[k%3],.4 if kind=='orth' else (.12,.4,.8)[k%3]); out=check((kind,k),a)
   if out: rows.append(out)
   else:
    with open('/tmp/l0_failed_'+kind+'_'+str(k)+'.pkl','wb') as f: pickle.dump(a,f)
  print(kind,'passed',len(rows),'max nodes/gap/sec',np.max(rows,axis=0) if rows else None,flush=True)
if __name__=='__main__': main()
