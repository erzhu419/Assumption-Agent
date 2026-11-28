from explore import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
folds=list(KFold(5,shuffle=True,random_state=23).split(np.arange(52)))+[(np.arange(52),np.arange(52,72))]
print('response range',np.min(y),np.max(y),'summary',s['train']['summary'],'telemetry',s['telemetry_vec']); print('calibration archive');

def rational(g,Y,i,j,mode,ridge):
 F=poly(g,2); Fi=F[i]; Z=aff(g); Yi=Y[i]; n=len(i); d=Z.shape[1]
 penalty=np.r_[np.ones(d)*.03,np.ones(F.shape[1]-d)]
 def solve(D):
  M=Fi/D[i,None]; return np.linalg.solve(M.T@M+ridge*np.diag(penalty),M.T@Yi)
 if mode=='full':
  def den(a):return np.exp(np.clip(F[:,d:]@a,-2,2))
  seeds=[np.zeros(F.shape[1]-d)]
 elif mode=='diag':
  def den(a):return 1+g**2@a
  seeds=[np.zeros(g.shape[1])+.03]
 else:
  def den(a):return 1+(Z@a)**2
  seeds=np.r_[np.eye(d)*.3,np.random.default_rng(42).normal(size=(5,d))*.35]
 def obj(a):
  D=den(a); q=solve(D); return np.r_[(Fi@q/D[i]-Yi),.025*a,np.sqrt(ridge*penalty)*q]
 best=None
 for a in seeds:
  p=least_squares(obj,a,bounds=(0,2) if mode=='diag' else (-2,2),max_nfev=100)
  if best is None or p.cost<best.cost:best=p
 D=den(best.x); q=solve(D); return F[j]@q/D[j],np.mean(abs(Fi@q/D[i]-Yi)),best.x
for mode in ['full','diag','ridge']:
 for label,g in [('x',x),('z',z),('h',h)]:
  for k in [.0001,.01]:
   es=[]
   for i,j in folds:
    p,tr,a=rational(g,y,i,j,mode,k); es.append(np.mean(abs(p-y[j])))
   print(mode,label,k,np.round(es,4),'inner',np.mean(es[:-1]),'train',tr,'param',a,flush=True)

def kernel(g,kind,i,j,Y):
 v=g/np.std(g[i],axis=0); t=v@v.T/4; D=(v[:,None]-v[None,:])**2/4; F=poly(g,2); base=F@F.T; n=len(i); a=Y[i]; I=np.eye(n)
 from scipy.linalg import cho_factor,cho_solve
 from scipy.optimize import minimize
 def get(b,ind):
  iv=np.ix_(ind,i); q=np.exp(b); out=base[iv]*2
  ve=D[iv]; tt=t[iv]; kd=ve@q[3:];
  if kind=='rbf': k=np.exp(-kd)
  elif kind=='matern': r=np.sqrt(3*kd+1e-20); k=(1+r)*np.exp(-r)
  elif kind=='cubic': k=tt**3
  else: k=np.exp(-kd)*(1+tt)**2
  return out+q[0]*k+q[1]*tt**3, k, ve, kd,tt
 def obj(b):
  K,k,ve,kd,tt=get(b,i); q=np.exp(b); C=cho_factor(K+q[2]*I); w=cho_solve(C,a); M=cho_solve(C,I)-np.outer(w,w); val=.5*a@w+np.log(np.diag(C[0])).sum()+.12*np.sum((b[3:]+1)**2)
  dk=-k if kind!='matern' else -1.5*np.exp(-np.sqrt(3*kd+1e-20));
  if kind=='cubic':dk=np.zeros_like(k)
  grad=.5*np.r_[np.sum(M*k),np.sum(M*tt**3),np.trace(M),np.einsum('ij,ijk->k',M*dk*q[0],ve)]*q; grad[3:]+=.24*(b[3:]+1)
  return val,grad
 p=minimize(obj,[-3,-6,-9,-1,-1,-1,-1],jac=True,bounds=[(-14,2)]*2+[(-13,-4)]+[(-5,3)]*4,method='L-BFGS-B')
 K=get(p.x,i)[0]+np.exp(p.x[2])*I; return get(p.x,j)[0]@cho_solve(cho_factor(K),a),p.x
for w,g in [('x',x),('z',z),('h',h),('Bz',z@B[1:]+B[0])]:
 for kind in ['rbf','matern','cubic','poly']:
  e=[]
  for i,j in folds:
   p,q=kernel(g,kind,i,j,y);e.append(np.mean(abs(p-y[j])))
  print('GP',w,kind,np.round(e,4),np.mean(e[:-1]),q,flush=True)
