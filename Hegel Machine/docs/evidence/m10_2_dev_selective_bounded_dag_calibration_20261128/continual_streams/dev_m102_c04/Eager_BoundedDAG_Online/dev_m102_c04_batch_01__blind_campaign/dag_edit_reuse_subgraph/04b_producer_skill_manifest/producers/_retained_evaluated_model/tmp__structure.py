exec(open('/tmp/routes.py').read().split('for typ in range(8):')[0])
from itertools import combinations_with_replacement as cwr,combinations
from scipy.optimize import least_squares
import runpy
basis=old['_temporal_basis'](s['eval_times'],t['traj'][:52]); Y=np.c_[t['y'],t['traj']@basis]
def rf(z,k):
 q=z*z; d=1+k[0]*q+k[1]*q*q; return np.c_[np.ones(len(q)),q/d,q*q/d]
F=lambda k:rf(u,k); k=least_squares(lambda k:(F(k)@ls(F(k),v)-v).ravel(),[.25,.04],bounds=(0,3),gtol=1e-12).x
r=lambda z:rf(z,k)@ls(F(k),v)
a,b=p1(x),r(x); dot=np.sum(a*b,axis=1)
def poly(d):return np.column_stack([np.prod(x[:,i],axis=1) for i in cwr(range(4),d)])
def outer(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
A=np.c_[a,b,dot]; blocks={'x':x,'P3':poly(3),'D3':np.column_stack([np.prod(x[:,i],axis=1) for i in combinations(range(4),3)]),'zp':outer(x,a),'zr':outer(x,b),'pr':outer(a,b),'P2':poly(2),'rr':outer(b,b),'pp':outer(a,a),'P4':poly(4),'zrdiag':x*b}
# Check if sign/parity of response is due to a small number of omitted terms.
for sets in [('x','zp','zr'),('x','P3','zr'),('x','D3','zr'),('D3',),('x','D3'),('x','D3','zrdiag'),('x','D3','pr'),('x','D3','pr','zrdiag'),('x','P2','P3','zr'),('x','P2','P3'),('x','zp','zr','pr'),('x','D3','zr','pr'),('x','D3','zr','P2')]:
 G=np.c_[A,*[blocks[q] for q in sets]]; mu=G[:52].mean(0); sd=np.maximum(G[:52].std(0),1e-10); G=(G-mu)/sd; Z=Y[:52].mean(0); U,S,V=np.linalg.svd(G[:52],full_matrices=False); H=G[52:]@V.T; runs=[]
 for l in (1e-8,1e-5,.001,.01,.03,.1,.3,1):
  W=(S/(S*S+l))[:,None]*(U.T@(Y[:52]-Z)); de=H@W+Z-Y[52:]; runs.append((l,np.mean(abs(de[:,0])),np.mean(abs(de[:,1:]@basis.T)),np.linalg.norm(G[:52]@V.T@W+Z-Y[:52],axis=0)))
 print(sets,'Y',min(runs,key=lambda q:q[1]),'T',min(runs,key=lambda q:q[2]))
# Rank individual terms after removing shared spatial contributions, using a multi-output sparse forward fit.
B=np.c_[*blocks.values()]; names=[f'{nm}{i}' for nm,b in blocks.items() for i in range(b.shape[1])]; n=52
for cols in ([0],[1,2],[1],[2]):
 C=affine(A); selected=[]
 for i in range(24):
  E=Y[:n,cols]-C[:n]@ls(C[:n],Y[:n,cols]); O=B[:n]-C[:n]@ls(C[:n],B[:n]); gain=np.sum((O.T@E)**2,axis=1)/np.maximum(np.sum(O**2,axis=0),1e-15); j=np.argmax(gain); selected.append(names[j]); C=np.c_[C,B[:,j]]; W=ls(C[:n],Y[:n,cols]); ep=C[n:]@W-Y[n:,cols];
  if i in [0,1,2,3,4,5,7,9,12,16,23]: print(cols,i+1,selected,np.mean(abs(ep),axis=0),np.std(E,axis=0))
