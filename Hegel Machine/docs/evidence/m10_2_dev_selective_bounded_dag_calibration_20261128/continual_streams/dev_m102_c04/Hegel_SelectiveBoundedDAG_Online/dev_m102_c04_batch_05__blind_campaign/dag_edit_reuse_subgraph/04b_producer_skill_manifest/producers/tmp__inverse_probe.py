exec(open('/tmp/spatial.py').read().split("x=s['train']['x']")[0])
from scipy.optimize import root
A=aop(s); B,C,D=ops(s)
b=s['panel_b']; w=ls(af(b['u']),b['v']);
def Bi(x):return np.linalg.solve(w[1:].T,(x-w[0]).T).T
def Ai(x):
 sols=[root(lambda t:A(t[None,:])[0]-r,np.clip(r,-1.5,1.5),tol=1e-10) for r in x];print('inverse max residual',max(np.linalg.norm(p.fun) for p in sols));return np.array([p.x for p in sols])
x=s['train']['x']; X={'x':x,'ai':Ai(x),'bi':Bi(x),'b':B(x),'a':A(x),'aibi':Ai(Bi(x)),'aib':Ai(B(x))};X.update({'biai':Bi(X['ai']),'bai':B(X['ai'])})
y=s['train']['y'];tr=s['train']['traj'];n=len(y); splits=[np.arange(52,n)]+list(np.array_split(np.random.default_rng(31).permutation(n),4)); folds=[(np.setdiff1d(np.arange(n),te),te) for te in splits]
def trial(f,t):
 e=np.array([np.abs(f[j]@ls(f[i],t[i])-t[j]).mean() for i,j in folds]);return np.mean(e),max(e),e[0],np.abs(f@ls(f,t)-t).mean()
F={};P={k:C(v) for k,v in X.items()};R={k:D(v) for k,v in X.items()}
for k,z in X.items():
 p,r=P[k],R[k];print(k, z.min(),z.max(), 'PC',trial(af(np.c_[p,r,p*r]),y))
 F[k,'rawpoly']=af(np.c_[z,z**2,z**3,z*cyclic(z)])
for kp,p in P.items():
 for kr,r in R.items():
  inter=(p[:,:,None]*r[:,None,:]).reshape(n,-1); off=inter[:,~np.eye(4,dtype=bool).ravel()]; p0=p-L if False else p
  for typ,j in [('lin',p[:,:0]),('dot',np.sum(p*r,axis=1,keepdims=True)),('diag',p*r),('pair',inter),('dot_off',np.c_[np.sum(p*r,axis=1),off])]:F[(kp,kr,typ)]=af(np.c_[p,r,j]);
for t in [y,tr]:
 print('top',t.shape); print(*sorted((trial(f,t),k) for k,f in F.items())[:12],sep='\n')
