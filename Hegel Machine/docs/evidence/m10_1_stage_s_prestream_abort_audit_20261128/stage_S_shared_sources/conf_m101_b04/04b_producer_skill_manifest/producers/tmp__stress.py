import importlib.util, pickle
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize, least_squares
from scipy.spatial.distance import cdist
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; I=np.arange(len(x))
r=np.linalg.norm(x,axis=1); w=s['test']['x']; edge=np.argsort(r)[-12:]; near=np.argsort(cdist(x,w).min(axis=1))[-12:]; corner=np.argsort(np.max(abs(x),axis=1))[-12:]
folds=[I[56:],*np.array_split(np.random.default_rng(121).permutation(len(x)),4),edge,near,corner]

def ard(a,b,T,kind,degree):
 F=m._poly(a,degree); G=m._poly(b,degree); Q,R=np.linalg.qr(F,mode='complete'); U=Q[:,F.shape[1]:]; Y=U.T@T; n=len(Y); q=T.shape[1]; D=(a[:,None,:]-a[None,:,:])**2
 def k(D,p):
  d=D@np.exp(p[:4]); return np.exp(-d/2) if kind=='gauss' else (1+d/2)**-2
 def calc(p):
  K=k(D,p); nug=np.exp(p[4]); C=U.T@K@U; C.flat[::n+1]+=nug; L=cholesky(C,lower=True); AY=cho_solve((L,True),Y); amp=np.mean(Y*AY); return (n*q*np.log(amp)/2+q*np.log(L.diagonal()).sum()+.7*np.sum((p[:4]-p[:4].mean())**2),AY,K,nug)
 p=minimize(lambda p:calc(p)[0],[-1.4]*4+[-8.],method='L-BFGS-B',bounds=[(-5,1)]*4+[(-15,-1)],options={'maxiter':80}).x
 _,AY,K,nug=calc(p); coef=U@AY; beta=np.linalg.lstsq(F,T-K@coef-nug*coef,rcond=None)[0]; return G@beta+k((b[:,None,:]-a[None,:,:])**2,p)@coef

def ridge(a,b,T):
 F=m._poly(a,2); G=m._poly(b,2); Q=np.linalg.qr(F,mode='reduced')[0]; V=T-Q@(Q.T@T); lam=1.e-4
 def H(X,p):
  P=p.reshape(3,5); return np.tanh(X@P[:,:4].T+P[:,4])
 def calc(p):
  A=H(a,p); B=A-Q@(Q.T@A); c=np.linalg.solve(B.T@B+lam*np.eye(3),B.T@V); return A,B,c
 def res(p):
  A,B,c=calc(p); return np.r_[(B@c-V).ravel(),np.sqrt(lam)*c.ravel(),.002*p]
 sol=min((least_squares(res,p,max_nfev=120) for p in np.random.default_rng(21).normal(0,.55,(8,15))),key=lambda s:s.cost); A,B,c=calc(sol.x); return G@np.linalg.lstsq(F,T-A@c,rcond=None)[0]+H(b,sol.x)@c

if __name__=='__main__':
 EY={}; EZ={}; families=[(k,d) for k in ('gauss','rq') for d in (1,2)]
 for v in folds:
  t=np.setdiff1d(I,v); a=x[t]; b=x[v]; st={'train':{'x':a,'y':y[t],'traj':z[t]},'test':{'x':b}}; base=m.solve_batch(st); modes=np.linalg.svd(z[t],full_matrices=False)[2][:2]; T=z[t]@modes.T
  ps={'base':base['y_hat']}; zs={'base':base['state_pred'],'ridge':ridge(a,b,T)@modes}
  for k,d in families:
   ps[k,d]=ard(a,b,y[t,None],k,d)[:,0]; zs[k,d]=ard(a,b,T,k,d)@modes
  for out,vals,Y in [(EY,ps,y[v]),(EZ,zs,z[v])]:
   for k,P in list(vals.items()):
    if k!='base':vals[k,'blend']=.5*(P+vals['base'])
   for k,P in vals.items():out.setdefault(k,[]).append(np.mean(abs(P-Y),axis=0))
 for out in EY,EZ:
  for k,E in out.items():
   E=np.asarray(E).reshape(len(folds),-1); print(k,'ordinary avg/max',np.round([E[:5].mean(),E[:5].max()],5),'stress max',np.round(E[5:].max(axis=1),5),'stress avg',np.round(E[5:].mean(axis=1),5))
