from graph import *
from scipy.linalg import solve
from scipy.optimize import minimize
A,AI=Amap(s['panel_a']); x=AI(s['train']['x']); y=np.c_[s['train']['y'],s['train']['traj']]; rng=np.random.default_rng(33); folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(4)]]
def eval(K,reg):
 out=[]
 for i,j in folds:
  r=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),y[i],assume_a='pos')-y[j]; out.append(np.mean(abs(r),0));
 return np.array(out)
d=x[:,None,:]-x[None,:,:]; D=d*d; R=[]
for l in [.12,.22,.35,.5,.65,.9,1.2,1.6,2.5]:
 for kind,k in [('rbf',np.exp(-D/(2*l*l))),('m32',(1+np.sqrt(3*D)/l)*np.exp(-np.sqrt(3*D)/l)),('m52',(1+np.sqrt(5*D)/l+5*D/(3*l*l))*np.exp(-np.sqrt(5*D)/l))]:
  for w in [0,.03,.1,.3,1,3,15]:
   K=1+x@x.T+k.sum(2)+w*sum(k[:,:,i]*k[:,:,j] for i in range(4) for j in range(i));
   for a in [.00003,.0003,.003,.015,.06]:
    r=eval(K,a); R.append((r.mean(0),r[0],kind,l,w,a))
for j in [0,8]:
 for r in sorted(R,key=lambda r:r[0][j])[:5]:print('GP',j,r)
# Exact dictionary including softsign, hinges and absolute values, with simple pair interaction blocks.
P=np.c_[np.ones(len(x)),x]; R=[]
for q in [0,.12,.22,.35,.5,.7,1,1.25,1.6,2,3,4,6]:
 for nm,f in [('abs',lambda u:abs(u-q)+abs(u+q)),('oddhinge',lambda u:np.maximum(u-q,0)-np.maximum(-u-q,0)),('bothhinge',lambda u:np.c_[np.maximum(u-q,0),np.maximum(-u-q,0)]),('logcosh',lambda u:np.logaddexp(q*u,-q*u)),('softsign',lambda u:u/(1+q*abs(u))),('tanh',lambda u:np.tanh(q*u)),('sin',lambda u:np.sin(q*u)),('poly',lambda u:np.c_[u*u,u**3])]:
  z=f(x)
  for p in [P,poly(x,2)]:
   for cross in [0,.1,1]:
    T=np.c_[p,z]; K=T@T.T+cross*((z@z.T)*(x@x.T)+(z@z.T)**2)
    r=eval(K,.001);R.append((r.mean(0),r[0],nm,q,p.shape[1],cross))
for j in [0,8]:
 for r in sorted(R,key=lambda r:r[0][j])[:10]:print('DICT',j,r)
for nm,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[nm]['u'],s[nm][j]; print(nm,j,flush=True); L=[]
 for label,f in [('softsign',lambda u,q:u/(1+q*abs(u))),('softpow',lambda u,q:u/(1+q[0]*abs(u)**q[1])),('cubeabs',lambda u,q:u**3/(1+q*abs(u))),('cpow',lambda u,q:u**3/(1+q[0]*abs(u)**q[1])),('signedpow',lambda u,q:np.sign(u)*abs(u)**q),('logcosh',lambda u,q:np.logaddexp(q*u,-q*u)),('hinge',lambda u,q:np.maximum(abs(u)-q,0)*np.sign(u))]:
  for fmode in [0,1,2]:
   def get(q,w):
    g=f(w,q if label in ('softpow','cpow') else q[0]); return np.c_[np.ones(len(w)),g] if fmode==0 else (np.c_[np.ones(len(w)),w,g] if fmode==1 else np.c_[np.ones(len(w)),w+q[-1]*g]);
   def cost(q):
    Z=get(q,u);return (Z@fit(Z,v)-v).ravel()
   n=2 if label in ('softpow','cpow') else 1; sol=least_squares(cost,[.8]*n+([.7] if fmode==2 else []),bounds=([.005]*n+([-30] if fmode==2 else []),[8]*n+([30] if fmode==2 else [])),max_nfev=200); L.append((np.std(sol.fun),label,fmode,sol.x));
 print(sorted(L)[:9]);
