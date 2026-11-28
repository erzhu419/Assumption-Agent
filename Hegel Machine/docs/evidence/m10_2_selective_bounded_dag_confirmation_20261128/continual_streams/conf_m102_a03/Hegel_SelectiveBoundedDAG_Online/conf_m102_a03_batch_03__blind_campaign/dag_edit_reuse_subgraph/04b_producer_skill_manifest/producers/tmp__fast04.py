exec(open('/tmp/probe04.py').read().split('for pn,kn')[0])
from scipy.linalg import solve
T=tr['traj'];U,S,V=np.linalg.svd(T[:52],full_matrices=False); Y=np.c_[tr['y'],T@V[:2].T]; scale=np.std(Y[:52],axis=0); A=a(x); e=Y-A@fit(A[:52],Y[:52]);
for j in (0,1,2):
 print('target',j,'affine',fit(A[:52],Y[:52,j]),flush=True)
 for k in range(4):
  idx=np.argsort(x[:52,k]); print(k,'bins',np.round([e[g,j].mean() for g in np.array_split(idx,13)],3),'stdev',np.round([e[g,j].std() for g in np.array_split(idx,13)],3),flush=True)
F=[*A.T]; names=['1','x0','x1','x2','x3'];
def add(v,n):F.append(v);names.append(n)
for j,z in enumerate(x.T):
 for k in range(j,4):add(z*x[:,k],f'x{j}x{k}')
 for f in np.arange(.5,12.01,.25):
  for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda t:t/(1+t*t))]:
   add(fn(f*z),f'{nm} {j} {f}')
 for b in np.arange(-.8,.81,.1):
  for nm,fn in [('step',lambda t:t>0),('hinge',lambda t:np.maximum(t,0))]:
   add(fn(z-b),f'{nm} {j} {b:.1f}')
 for k in range(4):
  if k!=j:
   for f in (1.,2.,3.,4.,5.,6.):
    for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:add(x[:,k]*fn(f*z),f'x{k}*{nm} {j} {f}')
F=np.array(F,float).T; fn=np.linalg.norm(F[:52],axis=0); fn[fn<1e-10]=1; F/=fn
for j in range(3):
 y=Y[:,j]; ix=list(range(5));
 for k in range(12):
  Q=np.linalg.qr(F[:52,ix])[0]; B=F[:52]-Q@(Q.T@F[:52]); err=y[:52]-Q@(Q.T@y[:52]); sc=abs(B.T@err)/np.maximum(np.linalg.norm(B,axis=0),1e-8);sc[ix]=-1; ix.append(np.argmax(sc)); yp=F[:,ix]@fit(F[:52,ix],y[:52]); print(j,k,names[ix[-1]],np.round([abs(yp[:52]-y[:52]).mean(),abs(yp[52:]-y[52:]).mean()],4),flush=True)
# Additive GP with coordinatewise smoothness selected by marginal likelihood; include interactions in second variant.
from scipy.optimize import minimize
from scipy.linalg import cho_factor,cho_solve
D=(x[:,None,:]-x[None,:,:])**2; lin=1+x@x.T
for j in range(3):
 for inter in (False,True):
  y=Y[:,j]; yy=y[:52]/scale[j]
  def cov(h):
   ell,amp,noi=np.exp(h[:4]),np.exp(h[4:8]),np.exp(h[8]); K=lin+np.sum(amp*np.exp(-.5*D/ell**2),axis=2)
   if inter: K+=np.exp(h[9])*np.exp(-.5*np.sum(D,axis=2)/np.exp(2*h[10]))
   return K,noi
  def obj(h):
   K,n=cov(h); C=cho_factor(K[:52,:52]+n*np.eye(52)); return .5*yy@cho_solve(C,yy)+np.log(np.diag(C[0])).sum()
  h=min((minimize(obj,np.r_[np.full(4,b),np.zeros(4),-5.,-2.,0.],bounds=[(-4,2)]*4+[(-10,4)]*4+[(-12,1),(-10,4),(-3,2)],method='L-BFGS-B') for b in (-2.,-1.,0.)),key=lambda r:r.fun).x
  K,n=cov(h); yp=K[52:,:52]@solve(K[:52,:52]+n*np.eye(52),yy,assume_a='pos')*scale[j]; print('GP',j,inter,np.round(h,2),abs(yp-y[52:]).mean(),flush=True)
