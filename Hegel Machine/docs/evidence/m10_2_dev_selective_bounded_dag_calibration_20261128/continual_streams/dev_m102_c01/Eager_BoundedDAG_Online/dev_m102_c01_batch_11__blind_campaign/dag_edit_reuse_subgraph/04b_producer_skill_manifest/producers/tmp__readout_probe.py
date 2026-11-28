import sys,pickle,numpy as np
from scipy.linalg import solve
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes')
import node_00 as a,node_02 as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; tr=s['train']['traj']; n=len(y)
full=a.run_node({'__input__':dict(s,test={'x':s['train']['x']})})
print('feature spectra',[(k, np.linalg.svd(full[k],compute_uv=False).round(3)) for k in ['sy','st']]);print('y std',y.std())
def poly(x,d):
 return np.column_stack([np.prod(x[:,idx],axis=1) for idx in combinations_with_replacement(range(x.shape[1]),d)])
def fold(i,j):
 p=a.run_node({'__input__':dict(s,train={k:s['train'][k][i] for k in ('x','y','traj')},test={'x':s['train']['x'][j]})}); pred=b.run_node({'node_00':p})['state_pred']; d={}; z,q=p['sy'],p['qy']; yy=y[i]; one=np.ones((len(i),1)); oq=np.ones((len(j),1));
 def test(name,f,g,K=None,Q=None):
  if K is None: K=f@f.T;Q=g@f.T
  for r in [1e-6,1e-4,.001,.01,.1,1.]:
   w=solve(K+r*np.eye(len(i)),yy,assume_a='pos'); d[name,r]=np.mean(abs(Q@w-y[j]))
 test('sy',np.c_[one,z],np.c_[oq,q]);test('sy2',np.c_[one,z,poly(z,2)],np.c_[oq,q,poly(q,2)]);test('sy3',np.c_[one,z,poly(z,2),poly(z,3)],np.c_[oq,q,poly(q,2),poly(q,3)]);test('kernel',None,None,p['K'],p['Kq']);
 for sk in [1,10,100,1000]: test('combined'+str(sk),None,None,p['K']+sk*(z@z.T),p['Kq']+sk*(q@z.T));
 for f,g,key in [(tr[i]@p['modes'].T,pred@p['modes'].T,'scores'),(p['st'],p['qt'],'st')]:
  test(key,np.c_[one,z,f],np.c_[oq,q,g]);test(key+'sy2',np.c_[one,z,poly(z,2),f],np.c_[oq,q,poly(q,2),g]);
 print('slice',j[0],np.mean(abs(pred-tr[j])),sorted(d.items(),key=lambda kv:kv[1])[:8]); return d
splits=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(102); splits += [(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(rng.permutation(n),4)]; ds=[fold(i,j) for i,j in splits]; print('by worst and mean:');
for s in sorted([(max(d[k] for d in ds),np.mean([d[k] for d in ds]),k,[round(d[k],5) for d in ds]) for k in ds[0]])[:30]:print(s)
