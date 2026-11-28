import numpy as np, pickle,sys,itertools
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00,node_02
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,t=[s['train'][k] for k in ('x','y','traj')]; n=len(y); ix=np.arange(n)
folds=[ix[-20:]]+list(np.array_split(ix,4))+[np.random.default_rng(i).permutation(ix)[:20] for i in range(3)]
def recover(K,Kq):
 n=len(K); A=K-4; l,U=eigh(A); X=U[:,-4:]*np.sqrt(l[-4:]/2); j,k=np.triu_indices(n); ri=np.arange(len(j)); jj=np.arange(4)
 def value(a,b,w):
  d=a-b; d2=d*d; m=a*b; D=m.sum(1); e=np.exp(-d2.sum(1)/6); f=np.exp(-(d2@w)/6)
  pred=1+D+.15*(D*D+(m*m).sum(1))+1.5*(e+f)
  g=(1+.3*D[:,None])*b+.3*m*b-.5*d*(e[:,None]+f[:,None]*w)
  h=(1+.3*D[:,None])*a+.3*m*a+.5*d*(e[:,None]+f[:,None]*w)
  return pred,g,h,-.25*f[:,None]*d2
 def calc(v,jac):
  f=v[:-4].reshape(-1,4); P,G,H,W=value(f[j],f[k],v[-4:]);
  if not jac:return P-K[j,k]
  M=np.zeros((len(j),len(v))); M[ri[:,None],4*j[:,None]+jj]=G; M[ri[:,None],4*k[:,None]+jj]+=H; M[:,-4:]=W; return M
 best=None
 for seed in range(3):
  Z=X if seed==0 else X@np.linalg.qr(np.random.default_rng(seed).normal(size=(4,4)))[0]; fit=least_squares(lambda v:calc(v,False),np.r_[Z.ravel(),np.ones(4)*.85],jac=lambda v:calc(v,True),bounds=(np.r_[np.full(4*n,-np.inf),np.full(4,.2)],np.r_[np.full(4*n,np.inf),np.full(4,3.)]),max_nfev=80,gtol=1e-10)
  if best is None or fit.cost<best.cost:best=fit
  if fit.cost<1e-14: break
 F=best.x[:-4].reshape(-1,4); w=best.x[-4:]; Q=[]
 for row in Kq:
  v=F[np.argmax(row)]; obj=lambda z:value(z[None,:],F,w); Q.append(least_squares(lambda z:obj(z)[0]-row,v,jac=lambda z:obj(z)[1],max_nfev=40).x)
 return F,np.array(Q),best.cost
if __name__=='__main__':
 data=[]
 for j in folds:
  i=np.setdiff1d(ix,j); p=node_00.run_node({'__input__':dict(s,train={'x':x[i],'y':y[i],'traj':t[i]},test={'x':x[j]})}); state=node_02.run_node({'node_00':p})['state_pred']; F,Q,cost=recover(p['K'],p['Kq']); print('recover',cost, np.max(np.abs(np.c_[np.ones(len(i)),F]@np.linalg.lstsq(np.c_[np.ones(len(i)),F],x[i],rcond=None)[0]-x[i])),flush=True); data.append((p,state,F,Q,y[j]))
 pickle.dump(data,open('/tmp/data10','wb'))
