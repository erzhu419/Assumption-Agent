from graph import *
import itertools as it,pickle
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; f,iv,op,c,d=maps(s); z=iv(t['x']); b=op(z); u=np.cos(2.6*z); H=aff(b); n=52; q=np.linalg.qr(H[:n])[0]; r=y[:n]-q@(q.T@y[:n]); pairs=list(it.combinations(range(4),2)); cols=[]; names=[]
def add(name,w):
 if np.ndim(w)==1 and np.isfinite(w).all():names.append(name);cols.append(w)
for k,x in [('b',b),('u',u),('c',c(b)),('d',d(b))]:
 base=[(f'{k}[:,{i}]',x[:,i]) for i in range(4)]; sums=base.copy(); prods=[]
 for i,j in pairs:
  a,A=base[i];h,B=base[j];prods.append((f'({a}*{h})',A*B));
  for sign in (1,-1):sums.append((f'({a}+({sign})*{h})',A+sign*B))
 prods+= [(f'({k}[:,{i}]*{k}[:,{j}]*{k}[:,{l}])',x[:,i]*x[:,j]*x[:,l]) for i,j,l in it.combinations_with_replacement(range(4),3)]
 for i,j,l,m in ((0,1,2,3),(0,2,1,3),(0,3,1,2)):
  for sign in (-1,1):prods.append((f'({k}[:,{i}]*{k}[:,{j}]+({sign})*{k}[:,{l}]*{k}[:,{m}])',x[:,i]*x[:,j]+sign*x[:,l]*x[:,m]))
 for a,A in sums+prods:
  add(a,A)
  for pow in (2,3):add(f'({a}**{pow})',A**pow)
  for j in (.22,.35,.5,.65,.8,1.,1.25,1.5,2.,2.15,2.5,3.):
   for fun,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:add(f'np.{fun}({j}*{a})',fn(j*A))
 for a,A in base+prods:
  for h,B in sums+prods:
   for j in (.1,.22,.35,.5,.75,1.25):add(f'({a}/(1+{j}*{h}**2))',A/(1+j*B**2))
A=np.array(cols).T.copy(); L=A[:n]-q@(q.T@A[:n]); l=np.linalg.norm(L,axis=0); L/=np.maximum(l,1e-10); cor=r@L; order=np.argsort(-abs(cor)); print('best singles:');
for j in order[:12]:
 h=np.c_[H,A[:,j]]; co=fit(h[:n],y[:n]);print(names[j],np.std(y[:n]-h[:n]@co),abs(y[n:]-h[n:]@co).mean(),co[-1])
print('best pairs:'); res=[]
for j in order[:200]:
 v=L[:,j]@L; sc=(cor-v*cor[j])**2/np.maximum(1-v*v,1e-8); m=np.argmax(sc); res.append((cor[j]**2+sc[m],j,m))
for _,j,k in sorted(res,reverse=True)[:15]:
 h=np.c_[H,A[:,[j,k]]];co=fit(h[:n],y[:n]);print(names[j],names[k],np.std(y[:n]-h[:n]@co),abs(y[n:]-h[n:]@co).mean(),co[-2:])
# Estimate hierarchical polynomial shrinkage using training-only PRESS.
import re
models=[]
for X,N in [(b,'b'),(u,'u')]:
 for cen in [0,1]:
  xx=X.copy();
  if cen:xx-=xx[:n].mean(0)
  xx/=xx[:n].std(0)
  for deg in [2,3,4,5,6]:
   B=np.column_stack([np.prod(xx[:,i],axis=1) for k in range(2,deg+1) for i in it.combinations_with_replacement(range(4),k)]); degrees=np.array([k for k in range(2,deg+1) for _ in it.combinations_with_replacement(range(4),k)]); B/=np.maximum(B[:n].std(0),1e-9)
   for scale in [.08,.15,.3,.5,.8,1.]:
    W=B*scale**(degrees-2); wr=W[:n]-q@(q.T@W[:n]); U,S,V=np.linalg.svd(wr,full_matrices=False)
    for lam in [.00001,.0001,.001,.01,.05,.1,.3,1.,3.]:
     d0=S*S/(S*S+lam); cv=(r-U@(d0*(U.T@r)))/(1-(q*q).sum(1)-(U*U)@d0); cc=V.T@(S/(S*S+lam)*(U.T@r)); yh=H[n:]@fit(H[:n],y[:n]-W[:n]@cc)+W[n:]@cc; models.append((abs(cv).mean(),abs(yh-y[n:]).mean(),N,cen,deg,scale,lam))
print('ridge best',sorted(models)[:15]);
# Profile a single rational or elementary nonlinear correction of a learned direction.
from scipy.optimize import least_squares
rows=[]
for X,N in [(b,'b'),(u,'u')]:
 for form in ('sin','tanh','rat','ratlin','ratquad','cos','power','soft'):
  def F(X,p):
   z=p[0]+X@p[1:5];
   if form=='sin':return np.sin(z)[:,None]
   if form=='cos':return np.cos(z)[:,None]
   if form=='tanh':return np.tanh(z)[:,None]
   if form=='soft':return np.logaddexp(0,z)[:,None]
   if form=='rat':return (1/(1+z*z))[:,None]
   if form=='power':return (z*np.abs(z)**1.25)[:,None]
   if form=='ratlin':return aff(X)/(1+z*z)[:,None]
   if form=='ratquad':return np.column_stack([X[:,i]*X[:,j] for i,j in it.combinations_with_replacement(range(4),2)])/(1+z*z)[:,None]
  def calc(p):
   a=np.c_[H,F(X,p)];co=fit(a[:n],y[:n]);return a@co-y
  for p in np.r_[np.c_[np.zeros(4),np.eye(4)],np.random.default_rng(9).normal(size=(9,5))]:
   opt=least_squares(lambda p:calc(p)[:n],p,bounds=(-12,12),max_nfev=90); e=calc(opt.x);rows.append((np.std(e[:n]),abs(e[n:]).mean(),N,form,np.round(opt.x,4).tolist()))
print('profile',sorted(rows)[:18]);
