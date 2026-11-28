import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
from sklearn.linear_model import lasso_path, orthogonal_mp
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; Y=tr['traj']; t=s['eval_times']; n=len(x)
def calibrate(p,x):
 u,v=p['u'],p['v']; r=lambda a:np.roll(a,-1,axis=1); F=lambda a:np.stack([np.ones_like(a),a,r(a),a*r(a),a**3],axis=2); D=F(u); keep=np.ones(len(u),bool)
 for _ in range(4):
  W=np.stack([np.linalg.lstsq(D[keep,i],v[keep,i],rcond=None)[0] for i in range(4)]); e=np.linalg.norm(np.einsum('nif,if->ni',D,W)-v,axis=1); keep=e<max(0.001,5*np.median(e))
 z=(x-W[:,0])/W[:,1]; eye=np.eye(4)
 for _ in range(15):
  j=(W[:,1]+W[:,3]*r(z)+3*W[:,4]*z*z)[:,:,None]*eye+(W[:,2]+W[:,3]*z)[:,:,None]*r(eye.T).T
  z-=np.clip(np.linalg.solve(j,(np.einsum('nif,if->ni',F(z),W)-x)[...,None])[...,0],-.5,.5)
 return z
z=calibrate(s['panel_a'],x)
def pm(k,l,f):
 p=s[k]; u=p['u']; D=lambda a,b:np.c_[np.ones(len(a)),f(a,b)]; fit=lambda b:np.linalg.lstsq(D(u,b),p[l],rcond=None)[0]; opt=minimize_scalar(lambda b:np.std(D(u,b)@fit(b)-p[l]),bounds=(.01,3),method='bounded').x; return D(z,opt)@fit(opt)
b=pm('panel_b','v',lambda a,k:a); c=pm('panel_c','v1',lambda a,k:a/(1+k*abs(a))); d=pm('panel_c','v2',lambda a,k:np.c_[a,1/(1+k*a*a)]); bases={'raw':x,'z':z,'b':b,'c':c,'d':d,'cz':c+z,'dz':d+z,'tanh':np.tanh(z)}
def Q(r):return -np.expm1(-t[:,None]*r)
def rates(T):
 return least_squares(lambda r:(Q(r)@np.linalg.lstsq(Q(r),Y[T].T,rcond=None)[0]-Y[T].T).ravel(),[.4,1.8],bounds=(.01,10)).x
ij=list(combinations(range(4),2)); ijk=list(combinations(range(4),3))
def feats(z,level):
 C=[]; names=[]
 def add(a,n):C.append(a);names.append(n)
 for i,u in enumerate(z.T):
  for p in range(1,level+3):add(u**p,f'{i}^{p}')
  if level>0:
   for nm,f in [('tanh',np.tanh),('sat',lambda u:u/(1+abs(u))),('rat',lambda u:1/(1+u*u)),('gauss',lambda u:np.exp(-u*u)),('atan',np.arctan),('sin',np.sin),('cos',np.cos),('sin2',lambda u:np.sin(2*u)),('cos2',lambda u:np.cos(2*u))]:add(f(u),f'{nm}{i}')
 for i,j in ij:
  u,v=z[:,i],z[:,j]
  for p in range(1,level+2):
   for q in range(1,level+3-p):add(u**p*v**q,f'{i}^{p} {j}^{q}')
  if level>0:
   add(u*v/(1+.5*(u*u+v*v)),f'rat{i}{j}');add(np.tanh(u)*np.tanh(v),f'tanhprod{i}{j}')
   for sign in [-1,1]:
    w=u+sign*v
    for k in [.7,1,1.5,2]:
     for nm,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda u:1/(1+u*u))]:add(f(k*w),f'{nm}{k}({i},{sign*j})')
 for idx in ijk:add(np.prod(z[:,idx],axis=1),str(idx))
 return np.column_stack(C),names
splits=[(np.arange(52),np.arange(52,72))]; splits += [(np.setdiff1d(np.arange(52),V),V) for V in np.array_split(np.random.default_rng(612).permutation(52),4)]
if __name__=='__main__':
 print('z bounds',z.min(0),z.max(0),'rates',rates(np.arange(52)), 'target std',y.std(),Y.std(0));
 for k,bs in bases.items():
  for level in (0,1):
   A,names=feats(bs,level); errs=[]; witnesses=[]
   for T,V in splits:
    q=Q(rates(T)); tar=np.c_[y,np.linalg.lstsq(q,Y.T,rcond=None)[0].T]; M=A[T].mean(0); S=A[T].std(0); F=(A[T]-M)/S; E=(A[V]-M)/S; center=tar[T].mean(0); U=tar[T]-center
    regs=np.geomspace(.00003,.1,14); ss=[]
    for i in range(3):
     _,w,_=lasso_path(F,U[:,i],alphas=regs[::-1],max_iter=3500,tol=.0006); o=orthogonal_mp(F,U[:,i],n_nonzero_coefs=16,return_path=True); ridge=np.stack([F.T@np.linalg.solve(F@F.T+r*np.eye(len(T)),U[:,i]) for r in [.001,.01,.1,1,3]],axis=1); ss.append(np.c_[w,o,ridge]);
    m=min(a.shape[1] for a in ss); P=[E@a[:,:m]+center[i] for i,a in enumerate(ss)]; em=np.abs(P[0]-y[V,None]).mean(0); tm=np.abs(P[1][:,None,:]*q[None,:,0,None]+P[2][:,None,:]*q[None,:,1,None]-Y[V,:,None]).mean((0,1)); errs.append([em,tm]); witnesses.append(ss)
   er=np.array(errs); ind=np.argmin(er[1:].mean(0),axis=1); print(k,level,'CV selected',ind,er[0,np.arange(2),ind].round(5),er[1:].mean(0)[np.arange(2),ind].round(5),'public min',er[0].min(1).round(5),flush=True)
   if k in ('z','b','c','d') and level==1:
    for i,W in enumerate(witnesses[0]):
     w=W[:,ind[i>0 and 1 or 0]]; jj=np.argsort(abs(w))[-8:][::-1]; print(i,[(names[j],round(w[j],3)) for j in jj])
