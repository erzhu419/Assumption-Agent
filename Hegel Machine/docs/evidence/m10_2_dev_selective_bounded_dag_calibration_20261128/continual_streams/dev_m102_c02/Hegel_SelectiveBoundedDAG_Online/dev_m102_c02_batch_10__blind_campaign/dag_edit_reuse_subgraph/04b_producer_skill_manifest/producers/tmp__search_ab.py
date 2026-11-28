exec(open('/tmp/fitmaps.py').read().split("tr=s['train']")[0])
from scipy.optimize import least_squares
from scipy.special import erf
fams={'rat':lambda z,a:z/(1+a*z*z),'atan':lambda z,a:np.arctan(a*z),'tanh':lambda z,a:np.tanh(a*z),'sin':lambda z,a:np.sin(a*z),'asinh':lambda z,a:np.arcsinh(a*z),'sqrt':lambda z,a:z/np.sqrt(1+a*z*z),'exp':lambda z,a:z*np.exp(-a*z*z),'erf':lambda z,a:erf(a*z),'cube':lambda z,a:z**3,'soft':lambda z,a:z/(1+a*np.abs(z))}
u,v=[s['panel_a'][k] for k in ('u','v')]; idx=np.delete(np.arange(len(u)),[31,5,17]); print('panel A',u.shape,u.min(0),u.max(0))
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; split=52

def cv(g):
 st=dict(s,train=dict(x=x[:split],y=y[:split],traj=t[:split])); o=sink.run_node({'__input__':st,'node_00':dict(c=g[:split],query=g[split:])}); return np.mean(abs(o['y_hat']-y[split:])),np.mean(abs(o['state_pred']-t[split:]))

def profile(f,lo=.03,hi=8):
 grid=np.geomspace(lo,hi,55); j=np.argmin([f(a) for a in grid]); return minimize_scalar(f,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded').x
rows=[]
for name,f in fams.items():
 for mode in range(4):
  def D(z,a,j):
   p=z[:,j]; q=z[:,(j+2)%4]; l=A(np.c_[p,q]); l=A(z) if mode==3 else l
   return np.c_[l,f(q,a)] if mode==0 else (np.c_[l,p**3,f(q,a)] if mode==1 else np.c_[l,f(p,a),f(q,a)])
  def fit(a):
   ds=[D(u[idx],a,j) for j in range(4)]; bs=[ls(d,v[idx,j]) for j,d in enumerate(ds)]; return np.array([d@b-v[idx,j] for j,(d,b) in enumerate(zip(ds,bs))]),bs
  a=profile(lambda a: np.mean(fit(a)[0]**2)); r,bs=fit(a); rows.append((np.mean(abs(r)),name,mode,a,max(abs(r.ravel())),bs))
U,V=[s['panel_b'][k] for k in ('u','v')]; BB=[]
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 U,V=[s[panel][k] for k in ('u',key)]; rr=[]
 for n,f in fams.items():
  for extra in range(3):
   def B(z,a): return np.c_[A(f(z,a)),z,z**2] if extra==2 else (np.c_[A(f(z,a)),z] if extra==1 else A(f(z,a)))
   a=profile(lambda a:errs(B(U,a),V)[1]); rr.append((errs(B(U,a),V)[1],n,extra,a,ls(B(U,a),V)))
 rr.sort(key=lambda r:r[0]); print(panel,key,[(e,n,m,a) for e,n,m,a,b in rr[:8]]); BB.append(rr[:4])
for e,n,m,a,emax,bs in sorted(rows,key=lambda r:r[0])[:12]:
 f=fams[n]
 def forward(z):
  p=z; q=np.roll(z,-2); out=[]
  for j in range(4):
   l=[1,*p] if m==3 else [1,p[j],q[j]]
   tail=[f(q[j],a)] if m==0 else ([p[j]**3,f(q[j],a)] if m==1 else [f(p[j],a),f(q[j],a)])
   out.append(np.dot(l+tail,bs[j]))
  return np.array(out)
 roots=[root(lambda z:forward(z)-p,p) for p in x]; z=np.array([r.x for r in roots]); scores=[]
 for rr in BB:
  for be,bn,bm,ba,bb in rr:
   F=fams[bn](z,ba); B=np.c_[A(F),z,z**2] if bm==2 else (np.c_[A(F),z] if bm==1 else A(F)); scores.append(cv(B@bb))
 print('A',e,n,m,a,emax,'inv',max(np.linalg.norm(r.fun) for r in roots),'z range',abs(z).max(),'scores',np.round(scores,4).tolist())
 if e==min(r[0] for r in rows):print('A best coef',bs);np.save('/tmp/z_best',z)
