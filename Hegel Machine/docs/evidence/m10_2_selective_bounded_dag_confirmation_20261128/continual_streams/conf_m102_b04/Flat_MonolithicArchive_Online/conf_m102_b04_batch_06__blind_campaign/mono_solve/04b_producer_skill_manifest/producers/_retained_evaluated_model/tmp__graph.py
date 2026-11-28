from probe import *
import importlib.util
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b04_batch_05_solver.py'); old=importlib.util.module_from_spec(sp); sp.loader.exec_module(old)
z=old._inverse(x,old._calibrate(s['panel_a'])); print('cal extrema',z.min(0),z.max(0)); print('temporal',np.linalg.svd(T,full_matrices=False)[2][:2]); print('delta temporal',np.linalg.svd(np.diff(T,axis=1),full_matrices=False)[2][:2]);
ops=[]
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; print(p,k); print('aff,cube',ls(np.c_[aff(u),u**3],v)); print('aff,sq',ls(np.c_[aff(u),u**2],v))
 def ff(u,m,c):
  if m==0:return np.c_[aff(u),u**3]
  if m==1:return np.c_[aff(u),np.sin(c*u)]
  if m==2:return np.c_[aff(u),np.sinh(c*u)]
  if m==3:return np.c_[aff(u),np.tanh(c*u)]
  if m==4:return np.c_[aff(u),np.expm1(-c*u*u)]
  if m==5:return np.c_[aff(u),u**2]
  if m==6:return np.c_[aff(u),u**3,u*np.roll(u,-1,axis=1)]
  if m==7:return np.c_[aff(u),u**3,u**5]
  if m==8:return np.c_[aff(u),u**3,u**2]
  if m==9:return aff(u+c*u**3)
  if m==10:return aff(np.sin(c*u))
  if m==11:return aff(np.tanh(c*u))
  if m==12:return np.c_[aff(u),u/(1+c*u*u)]
  if m==13:return np.c_[aff(u),u/np.sqrt(1+c*u*u)]
  if m==14:return np.c_[aff(u),np.log1p(c*u*u)]
 from scipy.optimize import minimize_scalar
 fits=[]
 for m in range(15):
  def score(c,ret=False):
   A=ff(u,m,c); w=ls(A,v); e=A@w-v; h=np.sum(np.linalg.qr(A,mode='reduced')[0]**2,axis=1); return np.mean(abs(e/(1-h[:,None]))) if not ret else w
  grid=np.linspace(.02,3,40); vs=[score(c) for c in grid]; i=np.argmin(vs); lo,hi=grid[max(0,i-1)],grid[min(39,i+1)]; o=minimize_scalar(score,bounds=(lo,hi),method='bounded'); fits.append((o.fun,m,o.x))
 print('panel loo',sorted(fits)[:8]); _,m,c=min(fits); w=score(c,True) if False else ls(ff(u,m,c),v); ops.append(lambda a,m=m,c=c,w=w:ff(a,m,c)@w)

def check(q):
 Y=np.c_[y,T]; out=[]
 def errs(pred):return np.mean(abs(pred-Y[52:]),axis=0).dot(np.vstack([np.r_[1,np.zeros(8)],np.r_[0,np.ones(8)/8]]).T)
 for d in (1,2,3):
  for g in [lambda h:h, np.tanh, lambda h:h/np.sqrt(1+.22*h*h),lambda h:np.sin(1.25*h)]:
   P=poly(g(q),d); scl=np.maximum(.01,np.sqrt(np.mean(P[:52]**2,axis=0))); P=P/scl; A=P[:52];
   for lam in (1e-6,.03,.3):
    E=errs(P[52:]@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@Y[:52]));out.append((E,d,lam,g.__name__))
 for gam in (.05,.15,.35,.8,1.6):
  ds=((q[:,None]-q[None,:])**2).sum(2); K=np.exp(-gam*ds)
  for lam in (1e-5,.003,.03):
   E=errs(Y[:52].mean(0)+K[52:,:52]@np.linalg.solve(K[:52,:52]+lam*np.eye(52),Y[:52]-Y[:52].mean(0)));out.append((E,'rbf',lam,gam))
 return sorted(out,key=lambda e:sum(e[0]))[:3]
from itertools import product
rows=[]
for base,X in [('x',x),('z',z)]:
 for n in range(3):
  for seq in product(range(3),repeat=n):
   q=X
   for j in seq:q=ops[j](q)
   rows.append((base,seq,check(q)))
for r in sorted(rows,key=lambda r:sum(r[2][0][0]))[:20]:print(r)
