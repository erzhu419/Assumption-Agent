import numpy as np
from scipy.optimize import root
from panels import s


def lin(x):
 return np.c_[np.ones(len(x)),x]

def a_terms(x,i):
 z=x[:,i]; w=x[:,(i+1)%4]
 return np.c_[np.ones(len(x)),z,w*(1+z),z**3]

def fit_a(p):
 u,v=p['u'],p['v']; w=np.ones(len(u)); M=[a_terms(u,i) for i in range(4)]
 for sc in (1,.1,.01,.001):
  for _ in range(8):
   b=np.stack([np.linalg.lstsq(m*w[:,None],v[:,i]*w,rcond=None)[0] for i,m in enumerate(M)])
   r=np.linalg.norm(v-np.column_stack([m@c for m,c in zip(M,b)]),axis=1)
   w=1/np.sqrt(1+(r/sc)**2)
 g=np.argsort(r)[:-3]
 return np.stack([np.linalg.lstsq(m[g],v[g,i],rcond=None)[0] for i,m in enumerate(M)]),r

def q(x):return x*np.roll(x,-1,axis=1)

def h(x):return x+np.logaddexp(1.1*x,-1.1*x)-np.log(2)

A,r=fit_a(s['panel_a']); B=np.linalg.lstsq(lin(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]; p=s['panel_c']; C=np.linalg.lstsq(lin(q(p['u'])),p['v1'],rcond=None)[0]; D=np.linalg.lstsq(lin(h(p['u'])),p['v2'],rcond=None)[0]
print('A',A,'residual norms',np.sort(r)[-5:]); print('C1/C2 residual',np.mean(np.abs(lin(q(p['u']))@C-p['v1'])),np.mean(np.abs(lin(h(p['u']))@D-p['v2'])))
def invert(x):
 z=np.array([root(lambda z:A[:,0]+A[:,1]*z+A[:,2]*np.roll(z,-1)*(1+z)+A[:,3]*z**3-xx,np.linalg.solve(np.diag(A[:,1])+A[:,2,None]*np.roll(np.eye(4),1,axis=1),xx-A[:,0]),tol=1e-10).x for xx in x]); print('invert err',np.max(np.abs(A[:,0]+A[:,1]*z+A[:,2]*np.roll(z,-1,axis=1)*(1+z)+A[:,3]*z**3-x)),'range',z.min(0),z.max(0)); return z

def forward(z): return lin(z)@B

def backward(z):return (z-B[0])@np.linalg.inv(B[1:])

x=s['train']['x']; z=invert(x); invert(s['test']['x']); b=forward(z); bi=backward(z)
Y=np.c_[s['train']['y'],s['train']['traj']]
F={'x':x,'z':z,'b':b,'bi':bi}
for k,w in list(F.items()):
 F[k+'1']=lin(q(w))@C; F[k+'2']=lin(h(w))@D
print('bases residuals train/test (y, traj)')
from itertools import combinations
out=[]
for n in (1,2,3,4):
 for keys in combinations(F,n):
  M=lin(np.column_stack([F[k] for k in keys])); beta=np.linalg.lstsq(M[:52],Y[:52],rcond=1e-9)[0]; e=abs(M@beta-Y); a,b=e[:52],e[52:]; out.append((max(b[:,0].mean(),b[:,1:].mean()),keys,a[:,0].mean(),a[:,1:].mean(),b[:,0].mean(),b[:,1:].mean()))
for r in sorted(out)[:24]:print(r[1],np.round(r[2:],6))
for metric in [2,3,4,5]:
 print('best by',metric,[(r[1],np.round(r[2:],5)) for r in sorted(out,key=lambda r:r[metric])[:5]])
