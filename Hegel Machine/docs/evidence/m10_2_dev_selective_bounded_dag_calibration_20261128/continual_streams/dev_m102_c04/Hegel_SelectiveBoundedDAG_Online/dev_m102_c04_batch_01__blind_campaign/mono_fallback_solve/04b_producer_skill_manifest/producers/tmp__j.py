from h import *
from scipy.linalg import qr
from itertools import combinations_with_replacement as cw
base={'q':q,'z':z,'Cq':C(q),'Cz':C(z),'Dq':D(q),'Dz':D(z),'DCq':D(C(q)),'DCz':D(C(z)),'CDq':C(D(q)),'CDz':C(D(z))}
for k,p in list(base.items()):
 if k in ('q','z','Cq','Cz','Dq','Dz'):
  base['B'+k]=B(p)

def terms(p,i):return np.column_stack([np.prod(p[:,j],axis=1) for j in cw(range(p.shape[1]),i)])
V=Y[:52]; U,sv,R=np.linalg.svd(T[:52],full_matrices=False);print('temporal SVD',sv,'basis',R[:4]); print('summary check',Y.mean(0)[[0,-1]],Y.std(0)[[0,-1]])
res=[]
def add(n,f):
 m=metrics(f);res.append((n,m))
for k,p in base.items():
 g=[terms(p,i) for i in range(1,6)];
 for i,j in [(1,3),(2,3),(2,4),(1,4),(0,3),(0,4),(0,5),(2,5)]:add((k,i,j),A(np.column_stack(g[i:j])))
 for l,r in base.items():
  if l==k:continue
  # Full cross interaction, with within-vector linear/quadratic terms as separate options.
  F=(p[:,:,None]*r[:,None,:]).reshape(-1,16)
  for extra in [p,r,np.c_[p,r],np.c_[poly(p,2)[:,1:],r],np.c_[p,poly(r,2)[:,1:]]]:add((k,l,'bilin',extra.shape[1]),A(np.c_[extra,F]))
  for a in (.22,1.25,2.4):
   for lab,h in [('div',p/(1+a*r*r)),('addcos',np.cos(a*(p+r))),('subcos',np.cos(a*(p-r))),('sindiff',np.sin(a*(p-r))),('cosprod',np.cos(a*p*r)),('divsum',p/(1+a*np.sum(r*r,1)[:,None])),('divdot',p/(1+a*np.sum(p*r,1)[:,None]**2)),('divprod',p/(1+a*(p*r)**2))]:
    add((k,l,lab,a),A(np.c_[p,r,h]));add((k,l,lab,a,'quadratic'),A(np.c_[poly(p,2)[:,1:],r,h]))
for j in [0,1]:
 print('TOP',j);print(*[(p,m[:,j].round(4)) for p,m in sorted(res,key=lambda kv:kv[1][1,j])[:16]],sep='\n')
# Characterize shared constant-rate dynamics versus input-dependent rates.
for k in range(1,5):
 def tb(rate):return -np.expm1(-t[:,None]*rate)
 def fun(rate):
  f=tb(rate);return (f@ls(f,T[:52].T)-T[:52].T).ravel()
 o=least_squares(fun,np.geomspace(.15,2.5,k),bounds=(.0001,12),max_nfev=300);print('rates',o.x,'MAE',np.mean(abs(o.fun)))
