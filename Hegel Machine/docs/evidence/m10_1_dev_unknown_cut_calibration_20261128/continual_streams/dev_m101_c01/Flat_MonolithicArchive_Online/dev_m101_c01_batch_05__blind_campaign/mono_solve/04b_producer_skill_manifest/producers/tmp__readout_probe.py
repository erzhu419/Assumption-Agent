from probe import *
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
I=np.arange(len(x)); splits=[I[56:],*np.array_split(np.random.default_rng(18).permutation(len(x)),4),np.argsort(np.linalg.norm(x,axis=1))[-16:]]
X=aff(x)@A; R=aff(X)@B; P=poly(R,2); D=cdist(R/R.std(0),R/R.std(0),'sqeuclidean')/4; K=P@P.T+.3*np.exp(-1.3*D)
def metrics(ep):return np.round(np.r_[ep[0],np.mean(ep,0),np.max(ep,0)],5)
def wave(p):return np.c_[np.exp(-p[0]*t)*np.sin(p[1]*t),-np.expm1(-p[2]*t)]
def prep(it):
 def res(p):
  f=wave(p); return (f@fit(f,z[it].T)-z[it].T).ravel()
 p=least_squares(res,[.2,1.1,.55],bounds=([0,.5,.08],[1,2,2])).x; b=wave(p); a=fit(b,z.T).T; return a,b,p
for base in ('kernel','quadratic'):
 for mode in ('direct','oracle','joint','state_quad','sparse_den','ridge_den'):
  E=[]
  for ids in splits:
   it=np.setdiff1d(I,ids); F=cho_factor(K[np.ix_(it,it)]+1e-5*np.eye(len(it))); H=K[np.ix_(ids,it)]; a,b,p=prep(it); ah=H@cho_solve(F,a[it]); S=ah@b.T
   if base=='quadratic': ah=P[ids]@fit(P[it],a[it]); S=ah@b.T
   if mode=='direct': yh=H@cho_solve(F,y[it]) if base=='kernel' else P[ids]@fit(P[it],y[it]);
   if mode in ('oracle','joint','state_quad'):
    at=a[ids] if mode=='oracle' else ah; w=np.c_[R,a]; wt=np.c_[R[ids],at]; gp=poly(w,2); gt=poly(wt,2)
    if mode=='state_quad':
     def small(a,r):return np.c_[poly(r,2),poly(a,2)[:,1:]]
     gp=small(a,R);gt=small(at,R[ids]);
    col=np.sqrt(np.sum(gp[it]**2,axis=0)); G=gp[it]/col; T=gt/col
    beta=np.linalg.solve(G.T@G+.0005*np.eye(G.shape[1]),G.T@y[it]); yh=T@beta
   if mode in ('sparse_den','ridge_den'):
    def one(w):
     def calc(h,which):
      den=1+(w[which]**2@np.exp(h)); return P[which]/den[:,None]
     def cost(h):
      g=calc(h,it); e=g@fit(g,y[it])-y[it];return e@e
     h=minimize(cost,[-3.]*w.shape[1],bounds=[(-12,0)]*w.shape[1],method='L-BFGS-B').x;g=calc(h,it); return calc(h,ids)@fit(g,y[it]),cost(h)
    if mode=='ridge_den':
     def fun(h):
      d=1+(R[it]@h[:4]+h[4])**2; g=P[it]/d[:,None]; return g@fit(g,y[it])-y[it]
     pf=min([least_squares(fun,ini,max_nfev=80) for ini in np.r_[np.eye(5)*.4,np.ones((1,5))*.1]],key=lambda rr:rr.cost);h=pf.x;yh=(P[ids]@fit(P[it]/(1+(R[it]@h[:4]+h[4])**2)[:,None],y[it]))/(1+(R[ids]@h[:4]+h[4])**2)
    else: yh,_=min([one(w) for w in (x,q,r,R)],key=lambda pair:pair[1]);
   E.append([abs(yh-y[ids]).mean(),abs(S-z[ids]).mean(),abs(S[:,-1]-z[ids,-1]).mean()]);
  print(base,mode,metrics(E))
a,b,p=prep(I);print('wave',p,'coefficient stats',a.mean(0),a.std(0)); print('channel correlations',np.corrcoef(np.c_[y,a].T));
for w in (x,q,r,R):
 f=poly(w,2); e=np.c_[y,a]-f@fit(f,np.c_[y,a]); print('quadratic residual cross',np.round(np.corrcoef(e.T),3));
