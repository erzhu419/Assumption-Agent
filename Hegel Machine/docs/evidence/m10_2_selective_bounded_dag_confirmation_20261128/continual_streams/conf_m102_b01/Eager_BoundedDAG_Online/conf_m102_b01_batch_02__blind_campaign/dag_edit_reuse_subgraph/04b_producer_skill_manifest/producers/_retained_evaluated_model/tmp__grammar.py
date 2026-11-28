import itertools,pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; tr=s['train']['traj']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]; n=len(y); d=np.arange(n); ids=list(itertools.combinations(range(4),2)); p=np.column_stack([w[:,i]*w[:,j] for i,j in ids]); q=np.c_[np.ones(n),p]; a=np.c_[q,w,w*w]; ls=lambda X,z:np.linalg.lstsq(X,z,rcond=None)[0]
from scipy.special import comb
from itertools import combinations_with_replacement as cwr
from collections import Counter

def mono(x,i):return np.prod(x[:,i],axis=1)
def matrix(x,degree):return np.column_stack([mono(x,i) for k in range(degree+1) for i in cwr(range(x.shape[1]),k)])
c=s['panel_c']; u=c['u']; print('panel C degree fits:');
for deg in range(1,5):
 A=matrix(u,deg); print(deg,A.shape,[np.round(np.std((A@ls(A,z)-z),axis=0),5) for z in (c['v1'],c['v2'])]);
print('times',s['eval_times'],'telemetry',s['telemetry_vec']);
terms=[i for k in range(1,7) for i in cwr(range(4),k) if k!=2 or len(set(i))<2]; F=np.column_stack([mono(w,i) for i in terms]);
folds=[d[52:],np.argsort(np.linalg.norm(w,axis=1))[-18:]]+[vi for seed in range(10) for vi in np.array_split(np.random.default_rng(seed).permutation(d),6)];
def select(t,base,depth):
 A=base.copy(); fs=[]; gains=[]
 for j in range(depth):
  R=y[t]-A[t]@ls(A[t],y[t]); G=F[t]-A[t]@ls(A[t],F[t]); scores=abs(R@G)/(np.linalg.norm(G,axis=0)+1e-6); scores[fs]=-1; j=np.argmax(scores); fs.append(j); gains.append(scores[j]); A=np.c_[A,F[:,j]]
 return A,tuple(terms[j] for j in fs),gains
for base in [q,a]:
 print('full shortlist',[(i,np.round(g,4)) for i,g in zip(*select(d,base,10)[1:])]);
 for depth in range(1,6):
  e=[]; hist=Counter()
  for v in folds:
   t=np.setdiff1d(d,v); A,ix,_=select(t,base,depth); hist.update([ix[0]]); e.append(np.mean(abs(A[v]@ls(A[t],y[t])-y[v])));
  print(base.shape[1],depth,np.round([*e[:2],np.mean(e[2:]),max(e[2:])],5),hist.most_common(3));
