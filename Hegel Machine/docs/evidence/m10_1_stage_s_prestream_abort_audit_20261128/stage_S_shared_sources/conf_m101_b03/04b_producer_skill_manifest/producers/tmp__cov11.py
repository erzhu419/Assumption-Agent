import sys,pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
sys.path.insert(0,'/holdout_task')
from batch_solver import _coordinates,_mode_fit,_predict
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; D=_coordinates(s,X); Y=np.c_[tr['y'],tr['traj']]; idx=np.arange(len(X)); near=cdist(X/X.std(0),X/X.std(0)); np.fill_diagonal(near,np.inf)
folds=[*np.array_split(np.random.default_rng(202).permutation(len(X)),6),idx[-16:],np.argsort(near.min(1))[-12:]]
def model(A,B,x,xt,y,mode):
 m,sd=A.mean(0),A.std(0); A,B=(A-m)/sd,(B-m)/sd
 m,sd=x.mean(0),x.std(0); x,xt=(x-m)/sd,(xt-m)/sd
 m,sd=y.mean(),y.std(); y=(y-m)/sd; n=len(y); eye=np.eye(n)
 def pair(A,B,x,xt):return ((A[:,None]-B)**2,(x[:,None]-xt)**2,(1+A@B.T/4)**2)
 Q=pair(A,A,x,x); P=pair(B,A,xt,x)
 def kern(p,Q):
  d,dx,p2=Q; w=np.exp(-2*p[:4]); r2=d@w
  if mode=='merged':rho=np.exp(p[5]);r2+=rho*dx.sum(-1)
  r=np.sqrt(r2); h=np.exp(-r); k=(1+r+r2/3)*h; g=(1+r)*h/3
  if mode=='poly':extra=np.exp(p[5])*p2;k+=extra
  elif mode=='merged':extra=-.5*g*rho*dx.sum(-1)
  elif mode=='product':
   rr=np.sqrt(5*d*w); ff=(1+rr+rr**2/3)*np.exp(-rr); k=ff.prod(-1); g=5*(1+rr)*np.exp(-rr)*k[:,:,None]/(3*ff)
  return k,g,w,extra if mode in ('poly','merged') else None
 def obj(p):
  k,g,w,j=kern(p,Q); nu=np.exp(p[4]); cf=cho_factor(k+nu*eye); a=cho_solve(cf,y); v=y@a/n; R=(cho_solve(cf,eye)-np.outer(a,a)/v)/2; delta=p[:4]-np.log(4.)
  d=Q[0]; g=g[:,:,None] if g.ndim==2 else g
  dp=np.einsum('ij,ijk,ijk,k->k',R,g,d,w)+.3*delta
  if j is not None: grad=np.r_[dp,nu*np.trace(R),np.sum(R*j)]
  else:grad=np.r_[dp,nu*np.trace(R)]
  return n*np.log(v)/2+np.log(cf[0].diagonal()).sum()+.15*(delta@delta),grad
 p0=[np.log(4.)]*4+[-9.]; bd=[(-1,4)]*4+[(-16,-3)]
 if mode in ('poly','merged'):p0 += [-3.];bd += [(-11,4)]
 p=minimize(obj,p0,jac=True,bounds=bd,method='L-BFGS-B',options={'maxiter':120}).x
 return m+sd*(kern(p,P)[0]@cho_solve(cho_factor(kern(p,Q)[0]+np.exp(p[4])*eye),y))
variants=['base','poly','merged','product']; allpred={k:[] for k in variants}
for h in folds:
 i=np.setdiff1d(idx,h); a,F=_mode_fit(s['eval_times'],tr['traj'][i]); out=np.c_[tr['y'][i],a]
 for k in variants:
  q=np.column_stack([_predict(D[i], D[h], y) if k=='base' else model(D[i],D[h],X[i],X[h],y,k) for y in out.T]);allpred[k].append(np.c_[q[:,0],q[:,1:]@F]);
for k in ['poly','merged','product']:allpred['base+'+k]=[(a+b)/2 for a,b in zip(allpred['base'],allpred[k])]
allpred['all']=[np.mean(x,0) for x in zip(*(allpred[k] for k in variants))]
for k,p in allpred.items():
 E=np.array([np.abs(u-Y[h]).mean(0) for u,h in zip(p,folds)])[:,[0,1,4,8]];print(k,'mean',E[:6].mean(0).round(4),'worst',E[:6].max(0).round(4),'tail',E[-2].round(4),'isolated',E[-1].round(4))
