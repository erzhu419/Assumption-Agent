import numpy as np,itertools
from scipy.special import erf
D=np.load('/tmp/train14.npz'); z,g,j,k,y=(D[a] for a in ('z','g','j','k','y')); n=len(y)
from numpy.linalg import lstsq

def ls(a,b):return lstsq(a,b,rcond=1e-9)[0]
def cv(a,lam=0):
 scores=[]
 for tr,te in [(np.arange(52),np.arange(52,n))]+[(p[20:],p[:20]) for p in [np.random.default_rng(s).permutation(n) for s in range(4)]]:
  if lam:
   m=a[tr].mean(0); s=np.maximum(a[tr].std(0),1e-8); aa=(a[tr]-m)/s; bb=(a[te]-m)/s; pred=bb@np.linalg.solve(aa.T@aa+lam*np.eye(aa.shape[1]),aa.T@(y[tr]-y[tr].mean(0)))+y[tr].mean(0)
  else: pred=a[te]@ls(a[tr],y[tr]);
  e=abs(pred-y[te]); scores.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.round([scores[0],np.max(scores,axis=0)],4).tolist()

def group(*a):return np.column_stack([np.ones(len(z)),*a]);
P=[z,g,j,k]; N=['z','g','j','k']; cols={};
for a,nm in zip(P,N):
 for i in range(4):cols[f'{nm}{i}']=a[:,i]
for ai,(a,na) in enumerate(zip(P,N)):
 for b,nb in zip(P[ai:],N[ai:]):
  for i in range(4):
   for jj in range(i if na==nb else 0,4):cols[f'{na}{i}*{nb}{jj}']=a[:,i]*b[:,jj]
for i in range(4):
 for w in (.6,1.25,2.15,2.4,3.0):
  t=z[:,i]*w
  for nn,f in [('sin',np.sin),('cos',np.cos),('erf',erf),('atan',np.arctan),('tanh',np.tanh),('rat',lambda t:t/(1+t*t)),('sp',lambda t:np.logaddexp(0,t)),('cube',lambda t:t**3)]:cols[f'{nn}{i}@{w}']=f(t)
for inds in itertools.chain(itertools.combinations(range(4),2),itertools.combinations(range(4),3),[(0,1,2,3)]):
 a=np.column_stack([P[i] for i in inds]); b=np.column_stack([a[:,i]*a[:,j] for i,j in itertools.combinations_with_replacement(range(a.shape[1]),2)]); print('group',''.join(N[i] for i in inds),cv(group(a)),[(l,cv(group(a,b),l)) for l in (.03,.3,2,10)])
for ai,(a,na) in enumerate(zip(P,N)):
 for b,nb in zip(P[ai:],N[ai:]):
  print('rolled',na+nb,[cv(group(z,g,j,k,a*np.roll(b,r,axis=1)),.04) for r in range(4)])
A=np.column_stack(list(cols.values())); names=list(cols); tr=np.arange(52);te=np.arange(52,n)
for yi in (0,8):
 for seed in ('all','z','none'):
  ids=list(range(16 if seed=='all' else 4 if seed=='z' else 0)); print('OMP',yi,seed)
  for step in range(19):
   B=group(A[:,ids]); res=A[tr]-B[tr]@ls(B[tr],A[tr]); r=y[tr,yi]-B[tr]@ls(B[tr],y[tr,yi]); rank=abs(r@res)/np.maximum(np.linalg.norm(res,axis=0),1e-8); rank[ids]=0; i=rank.argmax();ids.append(i)
   if step<10 or step%3==0:
    B=group(A[:,ids]); pred=B[te]@ls(B[tr],y[tr,yi]); print(names[i],round(np.sqrt(np.mean((B[tr]@ls(B[tr],y[tr,yi])-y[tr,yi])**2)),4),round(np.mean(abs(pred-y[te,yi])),4),end='; ')
  print()
