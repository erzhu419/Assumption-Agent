import pickle, numpy as np
from itertools import combinations_with_replacement
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from operators import panels,lin,ls
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; target=np.c_[y,t]; n=len(x)
print('eval_times',s['eval_times'],'telemetry',s['telemetry_vec']); print('test ranges',s['test']['x'].min(0),s['test']['x'].max(0))
ops=panels(s); A=ops['A'](np.r_[np.zeros((1,4)),np.eye(4)]); B=ops['B'](np.r_[np.zeros((1,4)),np.eye(4)]);ops['a']=lambda z:(z-A[0])@np.linalg.inv(A[1:]-A[0]);ops['b']=lambda z:(z-B[0])@np.linalg.inv(B[1:]-B[0]);
print('A',A,'B',B)
states={'X':x}
for k,f in ops.items():states[k]=f(x)
for k,z in list(states.items()):
 if k!='X':
  for l in ('B','b','C','D','a','A'):
   if l.lower()!=k.lower():states[k+l]=ops[l](z)
folds=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(123)
for _ in range(5):
 ids=rng.permutation(n);folds.append((ids[:52],ids[52:]))
def kern(p,d):
 p=p/np.maximum(np.sqrt(np.mean(p*p,axis=0)),1e-8);return (1+p@p.T/p.shape[1])**d
Kx=kern(x,2); E={};
for k,z in states.items():
 for tag,K in [('p1',kern(z,1)),('p2',kern(z,2)),('p3',kern(z,3)),('rbf',np.exp(-.2*cdist(z,z,'sqeuclidean')))]:
  for mix,M in [('single',K),('X2',K+Kx)]:
   for reg in [1e-5,2e-4,.002,.02]:
    err=[]
    for i,j in folds:
     pred=M[np.ix_(j,i)]@cho_solve(cho_factor(M[np.ix_(i,i)]+reg*np.eye(len(i))),target[i]);err.append(np.abs(pred-target[j]).mean(0))
    e=np.array(err);E[k,tag,mix,reg]=np.c_[e[:,0],e[:,1:].mean(1)]
for col in range(2):
 print('output',col)
 for k,e in sorted(E.items(), key=lambda a:a[1][:,col].mean())[:18]:print(k,np.round(e[:,col],5),np.round(e[:,col].mean(),5))
np.savez('/tmp/cvops.npz',**states)
