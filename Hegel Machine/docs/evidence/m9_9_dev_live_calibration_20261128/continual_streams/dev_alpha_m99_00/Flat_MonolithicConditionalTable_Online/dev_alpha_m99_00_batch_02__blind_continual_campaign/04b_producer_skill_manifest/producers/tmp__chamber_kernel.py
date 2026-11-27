import numpy as np
from scipy.optimize import least_squares
p=np.load('/tmp/chamber_probe.npz'); globals().update({k:p[k] for k in p.files}); t0=np.r_[0,t]; one=np.ones(len(g));
def rm(e): return np.sqrt(np.mean(e**2,axis=0));
def run(label,A):
 P=np.zeros_like(E)
 for l in np.unique(lab):
  s=lab!=l; P[~s]=A[~s]@np.linalg.lstsq(A[s],E[s],rcond=None)[0]
 B=np.linalg.lstsq(A,E,rcond=None)[0]; err=E-P
 print(label,'LOLO rate/curve/last',np.round([rm(err)[0],np.sqrt(np.mean(err[:,1:]**2)),rm(err)[-1]],5),'worst lab rate/curve',np.round(np.max([[rm(err[lab==l])[0],np.sqrt(np.mean(err[lab==l,1:]**2))] for l in np.unique(lab)],axis=0),5))
 return B
A=np.c_[one,x]; b=run('g',np.c_[one,g]); B=run('full feature direction',A); run('direction + memory',np.c_[A,h,z]); run('direction + interactions',np.c_[A,g*h,g[:,None]*z]);
print('g coefficient',b[1].round(5)); print('feature projection kernel',(theta[3:]@B[1:]/(theta[3:]@theta[3:])).round(5)); print('intercept',B[0].round(5));
U,S,V=np.linalg.svd(E,full_matrices=False)
for j in range(2):
 print('component', j,'time shape relative to final / norm', np.round(V[j]/np.linalg.norm(V[j]),5)); print('lab means & g slopes',[(int(l),np.linalg.lstsq(np.c_[one,g][lab==l],(U[:,j]*S[j])[lab==l],rcond=None)[0].round(4).tolist()) for l in np.unique(lab)])
# Fit additive one-mode drift shapes including observed derivative at zero.
def expq(k):return np.r_[k,-np.expm1(-k*t)]
T=np.r_[1,t]
for kind, fn in {
 'linear minus exponential':lambda q: np.c_[T,expq(np.exp(q[0]))],
 't times exponential gate':lambda q:np.c_[T,np.r_[1,t*np.exp(-np.exp(q[0])*t)]],
 'rational gate':lambda q:np.c_[T,np.r_[1,t/(1+np.exp(q[0])*t)]],
 'quadratic rational gate':lambda q:np.c_[T,np.r_[1,t/(1+np.exp(q[0])*t*t)]],
 'two transient exponentials':lambda q:np.c_[T,expq(np.exp(q[0])),expq(np.exp(q[1]))],
 }.items():
 def fit(mask):
  A1=A[mask]; Y=E[mask]; a1=np.linalg.lstsq(A1,Y,rcond=None)[0]
  def coeff(q):
   F=fn(q); return np.linalg.lstsq(F,a1.T,rcond=None)[0].T
  fit=least_squares(lambda q:(A1@coeff(q)@fn(q).T-Y).ravel(),np.log([.8,3.] if 'two' in kind else [.8]),bounds=(-4,4)); return fit.x,coeff(fit.x)
 q,b=fit(np.ones(len(g),bool)); P=np.zeros_like(E)
 for l in np.unique(lab):
  s=lab!=l; q1,b1=fit(s); P[~s]=A[~s]@b1@fn(q1).T
 err=E-P; print(kind,'decay',np.exp(q).round(4),'LOLO rate/curve',np.round([rm(err)[0],np.sqrt(np.mean(err[:,1:]**2))],5),'full time-basis projection RMSE',np.sqrt(np.mean((A@b@fn(q).T-A@B)**2)).round(6)); print('  first feature coefficients',b[:2].round(5))
