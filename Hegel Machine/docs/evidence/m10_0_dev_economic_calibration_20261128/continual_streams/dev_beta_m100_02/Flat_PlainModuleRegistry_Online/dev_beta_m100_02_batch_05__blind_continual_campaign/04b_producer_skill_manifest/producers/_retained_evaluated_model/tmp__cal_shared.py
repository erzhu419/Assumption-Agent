import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=np.max(abs(y),axis=1)<5; X,Y=x[good],y[good]

def poly(x,n):
 u=x[:,[2,3,0,1]]
 return np.stack([np.ones_like(x)]+[x**k for k in range(1,n+1,2)]+[u**k for k in range(1,n+1)],axis=-1)
def lin(a,y): return np.array([np.linalg.lstsq(a[:,j],y[:,j],rcond=1e-10)[0] for j in range(4)])
def pred(a,b): return np.einsum('nij,ij->ni',a,b)
def rms(r):return np.sqrt(np.mean(r*r))
folds=[(np.setdiff1d(np.arange(len(X)),t),t) for order in (np.arange(len(X)),np.argsort(np.max(abs(X),axis=1)),np.random.default_rng(11).permutation(len(X))) for t in np.array_split(order,5)]
for n in (5,7,9,11,13):
 a=poly(X,n); b=lin(a,Y); print('poly',n,rms(pred(a,b)-Y),max(rms(pred(a[te],lin(a[tr],Y[tr]))-Y[te]) for tr,te in folds)); print(np.round(b,7))

def fn(t,k,m):
 if m=='T': return np.tanh(k*t)
 if m=='E': return erf(k*t)
 if m=='A': return np.arctan(k*t)
 if m=='R': return t/(1+k*t*t)
 if m=='S': return t/np.sqrt(1+k*t*t)
 if m=='N': return np.sin(k*t)

def design(x,k,m):
 u=x[:,[2,3,0,1]]; a,b,c=k
 own={'1':lambda:[fn(x,a,m[0])],'L':lambda:[fn(x,a,m[0]),x],'C':lambda:[fn(x,a,m[0]),x,x**3],'N':lambda:[fn(x,a,m[0]),np.sin(b*x),x],'D':lambda:[fn(x,a,m[0]),fn(x,b,m[0])],'P':lambda:[fn(x,a,m[0]),x/(1+b*x*x),x],'G':lambda:[x/(1+a*x*x+b*x**4),x]}[m[1]]()
 cross={'Q':lambda:[u,u*u],'R':lambda:[u,u*u/(1+c*u*u)],'L':lambda:[u,u*u,u*u/(1+c*u*u)],'N':lambda:[u,np.sin(c*u),1-np.cos(c*u)],'K':lambda:[u,1-np.cos(c*u)],'T':lambda:[u,np.tanh(c*u),np.tanh(c*u)**2],'S':lambda:[u,u/(1+c*u*u),u*u/(1+c*u*u)],'P':lambda:[u,u**3,u**2,u**4],'G':lambda:[u,u*np.exp(-c*u*u),1-np.exp(-c*u*u)]}[m[2]]()
 return np.stack([np.ones_like(x),*own,*cross],axis=-1)
def fit(x,y,m):
 def f(k,ret=False):
  a=design(x,k,m); b=lin(a,y); return b if ret else (pred(a,b)-y).ravel()
 z=min((least_squares(f,k,bounds=([.005,.005,.005],[6,6,6]),max_nfev=90,ftol=1e-10,gtol=1e-11) for k in ([.8,.6,.5],[1.5,1.2,1.2])),key=lambda z:z.cost)
 return z.x,f(z.x,True),rms(z.fun)
rs=[]
for a in 'TEARSN':
 for b in '1LCNDPG' if a=='R' else '1LCNDP':
  for c in 'QRLNKTSPG':
   m=a+b+c; k,p,e=fit(X,Y,m); rs.append((e,m,k,p))
for e,m,k,p in sorted(rs)[:12]:
 vals=[]
 for tr,te in folds:
  a,b,_=fit(X[tr],Y[tr],m); vals.append(rms(pred(design(X[te],a,m),b)-Y[te]))
 print('shared',m,e,max(vals),'scales',k,flush=True)
 if m in [r[1] for r in sorted(rs)[:4]]: print(p,flush=True)
