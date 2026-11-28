from probe import *
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
Y=np.c_[y,T]; ids=np.arange(len(x)); folds=[ids[56:],*np.array_split(ids,4),ids[np.argsort(np.linalg.norm(x,axis=1))[-14:]]]
L=poly(x,1); P=poly(x,2); C=poly(x,3)[:,15:]
def rat(Z,ii,jj,mode,degree,ridge):
    X=P if degree==2 else np.c_[P,C]; Xi=X[ii]; Li=L[ii]; n=len(ii); v=Z[ii]; size=v.shape[1]; scale=np.linalg.norm(v,axis=0)/np.sqrt(n); v=v/scale
    def design(X,L,p):
        if mode=='quad': den=1+L[:,1:]@p[:4]+P_local(L)@p[4:]
        else:
            q=L@p; den=1+q*q if mode=='sq' else np.sqrt(1+q*q) if mode=='root' else np.exp(np.clip(q,-5,5))
        return X/np.maximum(den[:,None],.15)
    k=14 if mode=='quad' else 5
    reg=np.r_[np.full(15,1e-8),np.full(X.shape[1]-15,ridge)]; R=np.diag(np.sqrt(reg))
    def fit(D):
        return np.linalg.lstsq(np.r_[D,R],np.r_[v,np.zeros((len(R),size))],rcond=None)[0]
    def obj(p):
        D=design(Xi,Li,p); w=fit(D); return np.r_[(D@w-v).ravel(),(R@w).ravel(),.04*p]
    starts=[np.zeros(k)] if mode in ('quad','exp') else [np.r_[b,.4*d] for b in [0,.35] for d in [*np.eye(4),np.ones(4)/2]]
    best=min((least_squares(obj,p,bounds=(-2,2),max_nfev=90) for p in starts),key=lambda r:r.cost); w=fit(design(Xi,Li,best.x)); return design(X[jj],L[jj],best.x)@w*scale, best.x

def P_local(L):
    return np.column_stack([L[:,i]*L[:,j] for i in range(1,5) for j in range(i,5)])
def errors(pred,jj):
    e=abs(pred-Y[jj]);return [e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
for mode in ['quad','sq','root','exp']:
 for deg,ridge in [(2,0),(3,.08)]:
  for shared in [True,False]:
   E=[]; pp=[]
   for jj in folds:
    ii=np.setdiff1d(ids,jj); V=np.linalg.svd(T[ii],full_matrices=False)[2][:2]; Z=np.c_[y,T@V.T]; parts=[Z] if shared else [Z[:,:1],Z[:,1:]]; out=[rat(z,ii,jj,mode,deg,ridge) for z in parts]; F=np.hstack([r[0] for r in out]); pp.append([r[1].round(2).tolist() for r in out]); E.append(errors(np.c_[F[:,0],F[:,1:]@V],jj))
   E=np.array(E); print(mode,deg,shared,'avg',E.mean(0),'worst',E.max(0),'last16',E[0],'boundary',E[-1], 'params',pp[0],pp[-1],flush=True)
