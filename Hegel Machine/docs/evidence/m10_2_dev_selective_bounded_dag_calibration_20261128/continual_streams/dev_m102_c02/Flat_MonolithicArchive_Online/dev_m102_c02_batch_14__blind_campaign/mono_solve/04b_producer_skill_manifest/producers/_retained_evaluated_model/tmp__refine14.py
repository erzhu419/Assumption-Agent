import numpy as np
from scipy.optimize import least_squares
from fit14 import aff,fit
M=np.load('/tmp/m14.npz'); z,b=M['z'],M['b']; y=M['Y'][:,0]; q=np.sign(z)*np.maximum(abs(z)-.35,0)**1.5; bz=b*np.roll(z,-1,axis=1); g=bz.sum(1); n=52
spl=[np.arange(52,72),*np.array_split(np.random.default_rng(11).permutation(72),4),*[np.argsort(z[:,i])[-20:] for i in range(4)],*[np.argsort(z[:,i])[:20] for i in range(4)]]
ix=np.arange(len(y)); train=[np.setdiff1d(ix,t) for t in spl]; base=aff(np.c_[b,g]); h=np.c_[base,q/(1+.5*z[:,1:2]**2)]
def loo(h):
 u,s,v=np.linalg.svd(h[:n],full_matrices=False); u=u[:,s>s[0]*1e-9]; return np.mean(abs((y[:n]-u@(u.T@y[:n]))/np.maximum(1e-9,1-(u*u).sum(1))))
def met(h):
 a=[np.mean(abs(h[t]@fit(h[r],y[r])-y[t])) for r,t in zip(train,spl)];return np.round([loo(h),a[0],max(a[1:5]),max(a[5:]),np.mean(a[5:])],6)
print('BASE',met(h)); rows=[]
def add(k,D):rows.append((loo(D),k,D))
for l in [.22,.5,1,1.25,2.15]:
 for X,N in [(z,'z'),(q,'q'),(b,'b'),(bz,'bz')]:
  for j,x in enumerate(X.T):
   for label,G in [('div',1/(1+l*x*x)),('tanh',np.tanh(l*x)),('sin',np.sin(l*x)),('cos',np.cos(l*x))]:
    for nm,U in [('one',np.ones((len(y),1))),('b',b),('q',q),('g',g[:,None]),('bz',bz),('z',z)]:
     add((N,j,l,label,nm),np.c_[h,G[:,None]*U]);add(('replace',N,j,l,label,nm),np.c_[base,G[:,None]*U])
for name,U in [('b',b),('q',q),('z',z),('bz',bz),('g',g[:,None])]:
 for j,x in enumerate(U.T):
  for name2,V in [('b',b),('q',q),('z',z),('bz',bz),('g',g[:,None])]:
   for k,w in enumerate(V.T):add(('pair',name,j,name2,k),np.c_[h,x*w])
for _,k,D in sorted(rows,key=lambda u:u[0])[:24]:print(k,met(D))
print('sparse variants')
for l in [.22,.5,1.25,2.15]:
 for i in [0,1]:
  D=np.c_[base if i==0 else aff(np.c_[b,bz]),q/(1+l*z[:,1:2]**2)]; print(l,i,met(D))
print('strong tail contenders among leading held-in models')
print(*sorted([(met(D)[3],k,met(D)) for _,k,D in sorted(rows,key=lambda u:u[0])[:80]],key=lambda u:u[0])[:10],sep='\n')
