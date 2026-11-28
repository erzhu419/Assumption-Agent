exec(open('/tmp/probe.py').read().split("print('A row error'")[0])
from scipy.optimize import least_squares,minimize_scalar
from scipy.special import expit,erf
from itertools import combinations,combinations_with_replacement
N=52
print('ROWS',np.round(np.c_[c,u,y,t[:,-1]][np.argsort(y)][::2],2).tolist(),flush=True)
def aff(z): return np.c_[np.ones(len(z)),z]
def poly(z,d): return np.column_stack([np.prod(z[:,j],axis=1) for i in range(d+1) for j in combinations_with_replacement(range(z.shape[1]),i)])
def score(F):
 B=np.linalg.lstsq(F[:N],Y[:N],rcond=1e-8)[0]; E=abs(F@B-Y); return np.array([E[:N,0].mean(),E[N:,0].mean(),E[N:,1:].mean()])
zp=pc['u']; v=pc['v1'];
for kind, g in [('tanh',lambda z,k:np.tanh(k*z)),('exp',lambda z,k:z*np.exp(-k*z*z)),('erf',lambda z,k:erf(k*z))]:
 def f(z,k):return np.c_[aff(z),g(z,k)]
 def fit(k):
  F=f(zp,k); B=np.linalg.lstsq(F,v,rcond=None)[0];return np.std(F@B-v),B
 k=minimize_scalar(lambda k:fit(k)[0],bounds=(.1,4),method='bounded').x;sd,B=fit(k);print('panel-c1',kind,k,sd,flush=True)
 if kind=='tanh':v1=f(u,k)@B
f2=lambda z:np.c_[aff(z),z*z];v2=f2(u)@np.linalg.lstsq(f2(zp),pc['v2'],rcond=None)[0]
# Compare coordinates and structured pairwise nonlinearities with leakage-free inner PRESS.
coords={'c':c,'u':u,'v1':v1,'v2':v2,'rot':(c-bb[0])@np.linalg.inv(bb[1:5]),'v1c':np.c_[c,v1],'v2c':np.c_[c,v2]}
H=Y[:N]; W=np.array([1]+[.45]*8)
def loo(F):
 X=F[:N]; U,S,V=np.linalg.svd(X,full_matrices=False); keep=S>S[0]*1e-8; U=U[:,keep]; R=(U@(U.T@H)-H)/np.maximum(1e-6,1-(U*U).sum(1))[:,None];return np.mean(abs(R)*W)
for name,z in coords.items():
 D=aff(z); Q=poly(z,2); dim=z.shape[1]; cols=[]; labels=[]
 def add(g,l):
  g=np.asarray(g); cols.extend(g.T if g.ndim==2 else [g]); labels.extend([str(l)+':'+str(j) for j in range(g.shape[1])] if g.ndim==2 else [str(l)])
 if dim<=4:
  h=np.c_[z,*[z[:,i]+k*z[:,j] for i,j in combinations(range(dim),2) for k in [-2,-1,-.5,.5,1,2]]]
  norms=[('norm'+str(tuple(inds)),(z[:,inds]**2).sum(1)) for m in [2,3,4] for inds in combinations(range(dim),m)]
  for l,g in norms:
   for k in [.25,.5,1,2,4,8,16,32]:
    add(np.exp(-k*g),(l,'exp',k));add(1/(1+k*g),(l,'rat',k))
  for k in [.5,1,1.25,2,4,8,16]:
   for l,g in [('sin',np.sin(k*h)),('cos',np.cos(k*h)),('rat',1/(1+k*h*h)),('odd',h/(1+k*h*h)),('tanh',np.tanh(k*h)),('gauss',np.exp(-k*h*h))]:add(g,(l,k))
  add(poly(z,3)[:,Q.shape[1]:],'cubic'); add(poly(z,4)[:,poly(z,3).shape[1]:],'quartic')
 else:
  add(Q[:,D.shape[1]:],'quad')
 G=np.column_stack(cols); qrows=[]
 for base in ([D,Q] if dim<=4 else [D]):
  F=base.copy(); XX=F[:N]; Gs=G.copy(); chosen=[]; best=(loo(F),F,chosen)
  for i in range(20):
   U,S,_=np.linalg.svd(F[:N],full_matrices=False); U=U[:,S>S[0]*1e-8]; E=H-U@(U.T@H); J=Gs[:N]-U@(U.T@Gs[:N]); norm=np.linalg.norm(J,axis=0); v=np.linalg.norm((J.T@E)*W,axis=1)/np.maximum(norm,1e-12); v[norm<1e-6]=-np.inf
   top=np.argsort(v)[-12:]; k=min(top,key=lambda j:loo(np.c_[F,Gs[:,j]]));F=np.c_[F,Gs[:,k]];chosen=chosen+[labels[k]]; Gs[:,k]=0; cv=loo(F)
   if cv<best[0]:best=(cv,F,chosen)
  print('sparse',name,base.shape[1],best[0],score(best[1]),best[2],flush=True)
# Structured radial bump with center, diagonal scales, and low-degree background.
rng=np.random.default_rng(3)
for name,z in list(coords.items())[:5]:
 for bg in [1,2]:
  X=poly(z,bg)
  for kind in ['gauss','rational','tanh','softplus','cos','abs','logistic']:
   def F(w):
    r=((z-w[:4])**2)@np.exp(np.clip(w[4:8],-12,12)); a=w[8];
    g={'gauss':lambda:np.exp(-r),'rational':lambda:1/(1+r),'tanh':lambda:np.tanh(a-r),'logistic':lambda:expit(a-r),'softplus':lambda:np.logaddexp(0,a-r),'cos':lambda:np.cos(r+a),'abs':lambda:np.sqrt(r+.001)}[kind]();return np.c_[X,g]
   def resid(w):
    A=F(w)[:N]; R=(A@np.linalg.lstsq(A,H,rcond=1e-8)[0]-H)*W;return np.r_[R.ravel(),.003*w]
   opts=[least_squares(resid,w,max_nfev=100) for w in [np.zeros(9)]+list(rng.normal(0,1,(8,9)))]; best=min(opts,key=lambda a:a.cost);print('radial',name,bg,kind,round(best.cost,5),score(F(best.x)).round(5),best.x.round(3),flush=True)
