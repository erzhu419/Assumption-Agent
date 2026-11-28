import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit

def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def D(x):return np.c_[np.ones(len(x)),x]
def calibrate(panel,x):
 u,v=panel['u'],panel['v']; rng=np.random.default_rng(45); w=np.roll(u,2,axis=1)
 def gram(a,b,k):return D(np.c_[a,np.tanh(k[0]*a),np.sin(k[1]*b)])
 def coeff(k,g):return np.array([ls(gram(u[g,j],w[g,j],k),v[g,j]) for j in range(4)])
 def fore(z,k,c):return c[:,0]+z*c[:,1]+np.tanh(k[0]*z)*c[:,2]+np.sin(k[1]*np.roll(z,2,axis=1))*c[:,3]
 k=np.array([1.8,2.4]); g=np.arange(len(u)); a=[gram(u[:,j],w[:,j],k) for j in range(4)]
 c=min((coeff(k,i) for i in (rng.choice(len(u),9,False) for _ in range(100))),key=lambda c:np.median(np.max(abs(fore(u,k,c)-v),axis=1)))
 err=np.max(abs(fore(u,k,c)-v),axis=1); g=err<max(.001,6*np.median(err)); k=least_squares(lambda k:(fore(u[g],k,coeff(k,g))-v[g]).ravel(),k,bounds=([1.4,2],[2.2,2.8])).x; c=coeff(k,g)
 z=(x-c[:,0])/c[:,1]; j=np.arange(4)
 for i in range(12):
  J=np.zeros((len(z),4,4)); J[:,j,j]=c[:,1]+k[0]*c[:,2]*(1-np.tanh(k[0]*z)**2);J[:,j,(j+2)%4]=k[1]*c[:,3]*np.cos(k[1]*np.roll(z,2,axis=1));z-=np.linalg.solve(J,(fore(z,k,c)-x)[...,None])[...,0]
 return z

def panels(s,z):
 pb=s['panel_b']; bc=ls(D(pb['u']),pb['v']); b=D(z)@bc
 p=s['panel_c'];u=p['u']
 def S(a,k):return a/(1+k*abs(a))
 def L(a,k):return np.logaddexp(0,k*a)
 def cfit(fn,key,linear):
  def dm(x,k):return D(np.c_[x,fn(x,k)] if linear else fn(x,k))
  y=p[key];obj=lambda k:np.mean((dm(u,k)@ls(dm(u,k),y)-y)**2)
  k=minimize_scalar(obj,bounds=(.3,3),method='bounded').x
  return dm(z,k)@ls(dm(u,k),y),k,obj(k)**.5
 return b,cfit(S,'v1',True),cfit(L,'v2',False)

def trajectory(t,T):
 def basis(p):return -np.expm1(-t[:,None]*p)
 def err(p):
  H=basis(p);return (H@ls(H,T.T)-T.T).ravel()
 r=least_squares(err,[2.15,.4],bounds=([.8,.03],[5,1.3]),ftol=1e-12).x
 H=basis(r);return H,ls(H,T.T).T,r,np.mean(err(r)**2)**.5
