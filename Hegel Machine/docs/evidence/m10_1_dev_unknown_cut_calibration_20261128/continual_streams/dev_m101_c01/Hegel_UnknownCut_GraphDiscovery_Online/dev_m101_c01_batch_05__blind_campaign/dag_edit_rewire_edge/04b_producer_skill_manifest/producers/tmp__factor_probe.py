exec(open('/tmp/graph_probe.py').read().split("for key in ('v1','v2'):")[0])
from itertools import combinations_with_replacement
V=s['panel_c']['v1']; U=s['panel_c']['v2']; A=np.c_[np.ones(len(u)),u]
def fit(D,Y):return np.linalg.lstsq(D,Y,rcond=1e-10)[0]
def err(D,Y):return np.std(D@fit(D,Y)-Y)
for c in (1.7,2.,2.1,2.2,2.25,2.3,2.4):
 T=np.tanh(c*u); print('v1',c,'tanh only',err(np.c_[A[:,:1],T],V),'affine+tanh',err(np.c_[A,T],V))
for c in (1.25,1.3,1.35): print('v2',c,err(np.c_[A,np.cos(c*u)],U))
def resid(p):
 M=p[:16].reshape(4,4); h=p[16:]; D=np.c_[A,np.tanh(u@M+h)]; return (D@fit(D,V)-V).ravel()
p=least_squares(resid,np.r_[2.2*np.eye(4).ravel(),np.zeros(4)],max_nfev=150).x
print('tanhridge v1',np.std(resid(p)),p[:16].reshape(4,4),p[16:])
candidates=[('simple',2.2*np.eye(4),np.zeros(4)),('optimized',p[:16].reshape(4,4),p[16:])]
Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); rng=np.random.default_rng(18); folds=[np.arange(52,72)]+np.array_split(rng.permutation(n),4)
def cv(F,lam):
 out=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); mu=F[tr].mean(0); sd=np.maximum(F[tr].std(0),1e-8); D=(F-mu)/sd; H=D[tr]; K=H@H.T; pred=Y[tr].mean(0)+D[te]@H.T@np.linalg.solve(K+lam*np.eye(len(H)),Y[tr]-Y[tr].mean(0)); E=np.abs(pred-Y[te]);out.append([E[:,0].mean(),E[:,1:].mean()]);
 return np.array(out)
W={'raw':raw,'cal':x,'cal-B':x@b[1:]+b[0],'cal-invB':(x-b[0])@np.linalg.inv(b[1:]),'raw-B':raw@b[1:]+b[0],'raw-invB':(raw-b[0])@np.linalg.inv(b[1:])}
best=[]
def Q(z):return np.array([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)]).T
for pn,M,h in candidates:
 C1=fit(np.c_[A,np.tanh(u@M+h)],V); C2=fit(np.c_[A,np.cos(1.3*u)],U)
 for nm,w in W.items():
  a=np.c_[np.ones(len(w)),w]; z1=np.c_[a,np.tanh(w@M+h)]@C1; z2=np.c_[a,np.cos(1.3*w)]@C2
  for name,F in [('panel-linear',np.c_[z1,z2]),('panel-quad',np.c_[z1,z2,Q(np.c_[z1,z2])]),('panel-x',np.c_[w,Q(w),z1,z2,Q(z1),Q(z2),np.einsum('ij,ik->ijk',w,z1).reshape(len(w),-1)]),('coordinate-quad',np.c_[w,Q(w),np.tanh(w@M+h),np.cos(1.3*w),Q(np.tanh(w@M+h))])]:
   for lam in (.0001,.01,.1,1.):
    e=cv(F,lam); best.append((e[:,0].max(),e[0,0],e[:,1].max(),pn,nm,name,lam,e[:,0].round(4).tolist()))
for item in sorted(best)[:16]:print(item)
print('official best')
for item in sorted(best,key=lambda v:v[1])[:8]:print(item)
