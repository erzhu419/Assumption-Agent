import numpy as np,pickle
from itertools import combinations_with_replacement as cwr,combinations
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=S['train']['y']; Z=np.load('/tmp/features.npz'); n=len(y); z=Z['z'][:n]
rng=np.random.default_rng(26); folds=[np.arange(n-20,n)]+[rng.choice(n,20,False) for _ in range(9)]
def score(A):
 return np.array([np.mean(abs(A[j]@np.linalg.lstsq(A[i:=np.setdiff1d(np.arange(n),j)],y[i],rcond=1e-10)[0]-y[j])) for j in folds])
def rank(v):return .2*v[0]+.8*v[1:].mean()
def feats(z,d=3):return np.column_stack([np.ones(len(z))]+[np.prod(z[:,p],axis=1) for k in range(1,d+1) for p in cwr(range(4),k)])
out=[]
for p in np.linspace(-.1,.8,91):
 for kind in range(5):
  if kind==0:w=z/(1+p*z*z)
  elif kind==1:w=np.arctan(np.sqrt(max(.00001,p))*z)/np.sqrt(max(.00001,p))
  elif kind==2:w=z/np.sqrt(1+p*z*z)
  elif kind==3:w=z*np.exp(-p*z*z)
  else:w=z/(1+p*np.sum(z*z,axis=1,keepdims=True))
  A=feats(w)
  for typ,I in [('full',np.arange(35)),('no3way',[i for i,t in enumerate([()]+[p for k in (1,2,3) for p in cwr(range(4),k)]) if len(set(t))<3]),('noMix3',[i for i,t in enumerate([()]+[p for k in (1,2,3) for p in cwr(range(4),k)]) if len(t)<3 or len(set(t))==1])]:
   v=score(A[:,I]); out.append((rank(v),v[0],v[1:].max(),v[1:].mean(),kind,p,typ))
print('BEST:',*sorted(out)[:14],sep='\n');print('PUBLIC:',*sorted(out,key=lambda q:q[1])[:7],sep='\n')
A=feats(z/(1+.22*z*z)); tags=[()]+[p for k in (1,2,3) for p in cwr(range(4),k)]; coef=np.linalg.lstsq(A,y,rcond=None)[0];print('coeff',list(zip(tags,coef.round(3)))); print('residual',np.sort(abs(A@coef-y)).round(4))
