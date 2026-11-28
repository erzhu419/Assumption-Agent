from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
Q=poly(x,2)

def active(it,T,alpha):
 b=ls(Q[it],T)
 H=np.zeros((4,4,T.shape[1]))
 for (i,j),v in zip(combinations_with_replacement(range(4),2),b[5:]):
  H[i,j]+=v; H[j,i]+=v
 G=b[1:5][None,:,:]+np.einsum('ni,ijk->njk',x[it],H)
 G=G.reshape(-1,4) if T.shape[1]==1 else G.transpose(0,2,1).reshape(-1,4)
 M=G.T@G/len(G); return (1-alpha)*np.eye(4)+alpha*4*M/np.trace(M)

def run(it,iv,T,conf):
 alpha,l,noise,kind=conf
 M=active(it,T,alpha); eig,V=np.linalg.eigh(M); z=x@(V*np.sqrt(np.maximum(eig,0)))
 D=cdist(z,z,'sqeuclidean'); r=np.sqrt(D)/l
 K=np.exp(-r*r/2) if kind=='rbf' else (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
 A=Q[it]; K0=K[np.ix_(it,it)]+noise*np.eye(len(it)); Ka=K[np.ix_(iv,it)]
 Wi=np.linalg.solve(K0,np.c_[A,T]); p=ls(A.T@Wi[:,:15],A.T@Wi[:,15:]);return Q[iv]@p+Ka@(Wi[:,15:]-Wi[:,:15]@p)

from itertools import product
configs=list(product((0,.6,.95),(0.6,1.2,2.4),(.00001,.001,.03),('rbf','m52')))
for label,T in [('scalar',Y[:,:1]),('state',Y[:,1:])]:
 scores=[]
 for c in configs:
  allerr=[abs(run(it,iv,T[it],c)-T[iv]) for it,iv in folds]
  a=np.array([[e.mean(),e[:,-1].mean()] for e in allerr]); scores.append((c,a))
 print(label,'strongest fixed candidates')
 for c,a in sorted(scores,key=lambda m:m[1][:,1].max())[:5]:print(c,np.round(a.T,5))
 nested=[]
 for it,iv in folds:
  inner=[(np.setdiff1d(it,v),v) for v in np.array_split(np.random.default_rng(183).permutation(it),3)]
  w=min(configs,key=lambda c:np.mean([np.mean(abs(run(a,b,T[a],c)-T[b])) for a,b in inner]));e=abs(run(it,iv,T[it],w)-T[iv]);nested.append([e.mean(),e[:,-1].mean()]);
 print('nested',np.round(np.array(nested).T,5))
print('trajectory singular values',np.linalg.svd(j,compute_uv=False))
print('scalar target summary mean/std',y.mean(),y.std(),'state final',j[:,-1].mean(),j[:,-1].std())
