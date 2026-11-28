import itertools as it
import pickle, runpy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
f=runpy.run_path('/holdout_task/node_00.py'); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']
D=f['run_node']({'__input__':s}); n=len(D['y']); z=D['g'][:n,1:5]; x=s['train']['x']; b=D['fd'][:n]; c,d=np.hsplit(D['cd'][:n],2)
Y=np.c_[D['y'],D['traj']]; print('n, time, targets',n,s['eval_times'],np.std(Y,axis=0)); rng=np.random.default_rng(38); folds=[np.arange(n-20,n),*np.array_split(rng.permutation(n),5)]
def evaluate(G):
 E=[]
 for val in folds:
  ix=np.setdiff1d(np.arange(n),val); out=sink({'node_00':{'g':G[np.r_[ix,val]],'fd':b[np.r_[ix,val]],'cd':np.c_[c,d][np.r_[ix,val]],'y':Y[ix,0],'traj':Y[ix,1:]}})
  e=abs(Y[val]-np.column_stack((out['y_hat'],out['state_pred']))); E.append([e[:,0].mean(),e[:,1:].mean()])
 return np.round(E[0],5),np.round(np.max(E,axis=0),5)
def col(*xs):return np.column_stack(xs)
def powers(Q,deg,onlyadd=False):
 return col(*[np.prod(Q[:,ids],axis=1) for k in range(2,deg+1) for ids in it.combinations_with_replacement(range(Q.shape[1]),k) if not onlyadd or len(set(ids))==1])
base=col(np.ones(n),z,b,c,d)
for nm,Q in [('x',x),('z',z),('b',b),('c1',c),('c2',d),('panels',col(b,c,d))]:
 for deg in (2,3,5):
  if Q.shape[1]>4 and deg==5:continue
  G=col(base,powers(Q,deg,deg==5)); print(nm,deg,evaluate(G))
for nm,Q in [('z',z),('z b c d',col(z,b,c,d)),('b c d',col(b,c,d))]:
 Q=(Q-Q.mean(0))/Q.std(0); r=cdist(Q,Q,'sqeuclidean')/Q.shape[1]
 for w in [.15,.3,.6,1.2,2.4,4.8]:
  K=np.exp(-w*r); ev,V=np.linalg.eigh(K); G=V*np.sqrt(np.maximum(ev,0)); print('kernel',nm,w,evaluate(col(base,G)))
for nm,p,k in [('B',s['panel_b'],'v'),('C1',s['panel_c'],'v1')]:
 u,v=p['u'],p[k]; l=col(np.ones(len(u)),u)
 def loo(G):
  q=np.linalg.qr(G,mode='reduced')[0]; R=v-q@(q.T@v); return np.mean(abs(R/(1-(q*q).sum(1))[:,None])),np.mean(abs(R))
 atoms={**{f'sin{k}':np.sin(k*u) for k in (1,1.5,2,2.5,3,3.5,4)},**{f'tanh{k}':np.tanh(k*u) for k in (1,1.25,1.5,2,2.25,2.5,3)},**{f'rat{k}':u/(1+k*u*u) for k in (.5,1,2,3)},'quad':u*u,'cubic':u**3,'cos':np.cos(2.5*u),'abs2':u*abs(u)}
 ranked=sorted((loo(col(l,atoms[a],atoms[b])),a,b) for a,b in it.combinations(atoms,2)); print(nm,'two families',ranked[:8]);
 for a in [*(q[1] for q in ranked[:1]),'sin3','tanh2.25']:
  for kind,X in [('pair',powers(u,2)),('oddpair',col(*[u[:,i]**2*u[:,j] for i in range(4) for j in range(4) if i!=j])),('addquad',u*u)]:
   print(nm,a,kind,loo(col(l,atoms[a],X)))
