from probe import *
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import least_squares

def affine(u,v):
 return fit(poly(u,1),v)

def robust(u,v):
 X=poly(u,1); B=fit(X,v)
 for j in range(8):
  e=np.linalg.norm(v-X@B,axis=1); keep=np.argsort(e)[:len(e)-3]; B=fit(X[keep],v[keep])
 return B,keep

p=s['panel_a']; A,I=robust(p['u'],p['v']); u=p['u'][I]; v=p['v'][I]; print('A fit',A,'sv',np.linalg.svd(A[1:],compute_uv=False),'resid',np.sort(np.linalg.norm(v-poly(u,1)@A,axis=1)),'outliers',np.setdiff1d(np.arange(48),I))
for d in (1,2,3):
 X=poly(u,d); e=v-X@fit(X,v); h=np.sum(np.linalg.qr(X,mode='reduced')[0]**2,axis=1); print('A clean poly',d,np.mean(abs(e/(1-h[:,None])),axis=0))
for name in ('panel_b','panel_c'):
 p=s[name]; u=p['u']
 for k in p:
  if k=='u': continue
  v=p[k]; L=affine(u,v); print(name,k,'sv',np.linalg.svd(L[1:],compute_uv=False),'max abs',np.abs(v).max())
  for label, ux in [('u',u),('sin(u)',np.sin(u)),('tanh(u)',np.tanh(u)),('u,u^2',np.c_[u,u*u]),('u,sin(u)',np.c_[u,np.sin(u)]),('u,tanh(u)',np.c_[u,np.tanh(u)])]:
   X=poly(ux,1); e=v-X@fit(X,v); h=np.sum(np.linalg.qr(X,mode='reduced')[0]**2,axis=1); print(' ',label,np.mean(abs(e/(1-h[:,None]))))
  if name!='panel_c': continue
  for j in range(4):
   y=v[:,j]; a=fit(poly(u,1),y); vals=[]
   for label,f in [('tanh',np.tanh),('sin',np.sin),('cubic',lambda p:p**3),('sat',lambda p:p/(1+p*p)),('square',lambda p:p*p)]:
    def pred(x,b):
     return b[0]+b[1]*f(b[2]+x@b[3:])
    b0=np.r_[a[0],1.35,0,a[1:]/1.35]; vals.append((label,*[float(np.mean(abs(pred(u[k:],b.x)-y[k:]))) for k in (18,24) for b in [least_squares(lambda b:pred(u[:k],b)-y[:k],b0,max_nfev=120)] if k<24]))
   print(' single-index vj',j,vals)
t=s['eval_times']; z=s['train']['traj']; _,_,V=np.linalg.svd(z,full_matrices=False); print('time',t,'V',V[:2]);
def loss(D):
 return np.sqrt(np.mean((z-z@D@np.linalg.pinv(D))**2))
funcs={'t':lambda t,a:t,'exp':lambda t,a:-np.expm1(-a*t),'sin':lambda t,a:np.sin(a*t),'cos':lambda t,a:1-np.cos(a*t),'rise':lambda t,a:t*np.exp(-a*t),'t-exp':lambda t,a:t+np.expm1(-a*t)/a,'tanh':lambda t,a:np.tanh(a*t)}
from itertools import combinations_with_replacement
vals=[]
for a,b in combinations_with_replacement(funcs,2):
 def des(p):return np.c_[funcs[a](t,p[0]),funcs[b](t,p[1])]
 def res(p):
  D=des(p); return (z-z@D@np.linalg.pinv(D)).ravel()
 p=min((least_squares(res,p,bounds=(0.001,5),max_nfev=75) for p in [(0.8,1.5),(2.,.4)]),key=lambda r:r.cost)
 vals.append((loss(des(p.x)),a,b,p.x))
print('temporal fits',*sorted(vals,key=lambda v:v[0])[:12],sep='\n')
