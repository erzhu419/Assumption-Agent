from compose_probe import *
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
T=np.c_[y,z]; n=52
p=s['panel_a'];u,v=p['u'],p['v'];keep=np.linalg.norm(v,axis=1)<5
basis=lambda x:np.stack([np.ones_like(x),x,x**3,(x-1)*np.roll(x,-1,axis=1)],axis=2)
a=basis(u[keep]);w=np.stack([fit(a[:,i],v[keep,i]) for i in range(4)]);print('calibration weights',w,'cal residual',np.std((a*w).sum(2)-v[keep]));f=(basis(raw)*w).sum(2)
u=s['panel_c']['u']; print('C1 error',np.std(bil(u)@fit(bil(u),s['panel_c']['v1'])-s['panel_c']['v1']));print('C2 weights',np.round(fit(cos(u,2.4),s['panel_c']['v2']),4))
Dct={'raw':raw,'inverse':x,'forward':f}
for tag,g in list(Dct.items()):
 Dct[tag+'B']=B(g)
for tag,g in list(Dct.items()):
 c,d=C(g),D(g); h=bias(np.c_[c,d]); p=c*d
 for k,H in [('linear',h),('matched',np.c_[h,p]),('full',np.c_[h,(c[:,:,None]*d[:,None,:]).reshape(len(g),-1)]),('cross',np.c_[h,g*np.roll(c,1,axis=1)]),('poly',np.c_[h,poly(g,3)[:,5:]])]:
  R=H@fit(H[:n],T[:n])-T;print(tag,k,np.round([np.std(R[:n]),np.mean(abs(R[n:,0])),np.mean(abs(R[n:,1:]))],5))
def gp(F,H,mode,nt,idx=None,te=None):
 if idx is None: idx=np.arange(nt);te=np.arange(nt,len(F))
 F=F/F[idx].std(0); Q=(F[:,None]-F[None,idx])**2; I=np.eye(nt)
 X=H[idx]; P=np.linalg.svd(X,full_matrices=True)[0][:,np.linalg.matrix_rank(X):]; v=P.T@T[idx]; U,S,V=np.linalg.svd(v,full_matrices=False); G=U[:,:3]*np.sqrt(nt)
 def model(p,ret=False):
  dist=Q@np.exp(p[:-1]);
  if mode==0: K=np.exp(-dist)
  if mode==1:
   q=np.sqrt(3*dist);K=(1+q)*np.exp(-q)
  if mode==2: K=(1+dist)**-2
  K0=K[idx]+np.exp(p[-1])*I
  if ret:
   L=cho_factor(K0); Z=cho_solve(L,X); W=fit(X.T@Z,Z.T@T[idx]);return H[te]@W+K[te]@cho_solve(L,T[idx]-X@W)
  L=cho_factor(P.T@K0@P);scale=np.sum(G*cho_solve(L,G),axis=0)/len(P.T);return 2*np.log(np.diag(L[0])).sum()+len(P.T)*np.log(scale).mean()
 p=min((minimize(model,np.r_[np.full(F.shape[1],g),-7],method='L-BFGS-B',bounds=[(-9,4)]*F.shape[1]+[(-13,-1)],options={'maxiter':100}) for g in [-2,0]),key=lambda r:r.fun).x
 R=model(p,True)-T[te];return np.array([np.mean(abs(R[:,0])),np.mean(abs(R[:,1:]))]),np.exp(p)
if __name__=='__main__':
 folds=[(np.arange(n),np.arange(n,72))]+[(p[:n],p[n:]) for p in [np.random.default_rng(k).permutation(52) for k in range(2)]]
 for k,F in Dct.items():
  for tag,g in [('self',F),('inverse',x)]:
   if tag=='inverse' and k=='inverse':continue
   for mode in range(3):
    H=bias(np.c_[C(g),D(g)]);scores=[]
    for j,(idx,te) in enumerate(folds):
     if j:idx,te=idx[:40],idx[40:]
     e,p=gp(F,H,mode,len(idx),idx,te);scores.append(e)
    print('GP',k,tag,mode,np.round(scores,5).tolist(),flush=True)
