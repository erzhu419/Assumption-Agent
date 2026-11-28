from model14 import *
from itertools import combinations_with_replacement
f=np.load('/tmp/graph14.npz');z,b,p,q,Y,H=(f[k] for k in ('z','b','p','q','Y','H')); n=52
F={}
def add(name,a): F[name]=np.asarray(a).reshape(len(z),-1)
for k,a in dict(z=z,b=b,p=p,q=q,pplusq=p+q,pminusq=p-q,pq=p*q).items():
 add(k,a)
 for f,fn in fun.items():
  for t0 in (.22,1,2.15):add(f'{f}({t0}{k})',fn(t0*a))
for (k,a),(l,d) in combinations_with_replacement(dict(z=z,b=b,p=p,q=q).items(),2):
 add(k+l,a*d);add(k+'x'+l,a[:,:,None]*d[:,None,:]);add(k+'dot'+l,(a*d).sum(1));add(k+'rat'+l,a/(1+.22*d*d));add(l+'rat'+k,d/(1+.22*a*a));add('sin('+k+l+')',np.sin(2.15*a*d));add('tanh('+k+l+')',np.tanh(2.15*a*d))
def score(A,Y):
 a=A[:n]; bh=ls(a,Y[:n]); h=np.sum(a*np.linalg.pinv(a).T,axis=1); e=(a@bh-Y[:n])/(1-h.clip(max=.999)); return np.mean(e**2)**.5,np.mean(abs(A[n:72]@bh-Y[n:])),np.mean(abs(a@bh-Y[:n]))
for i,y in enumerate(Y.T):
 print('TARGET',i,'best single',sorted((score(D(a),y),k) for k,a in F.items())[:9]); keys=list(F); A=np.ones((len(z),1)); sel=[]
 for j in range(8):
  k=min(keys,key=lambda k:score(np.c_[A,F[k]],y)[0]);A=np.c_[A,F[k]];sel.append(k);keys.remove(k); print(j,sel,score(A,y))
