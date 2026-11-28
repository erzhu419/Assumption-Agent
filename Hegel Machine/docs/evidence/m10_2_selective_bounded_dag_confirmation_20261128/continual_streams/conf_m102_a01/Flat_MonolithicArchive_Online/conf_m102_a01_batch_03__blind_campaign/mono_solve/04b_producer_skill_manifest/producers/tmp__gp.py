import sys,pickle,numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations_with_replacement as comb
sys.path.insert(0,'/holdout_task'); import batch_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x']; y=s['train']['y'];T=s['train']['traj'];z,h=old._coordinates(s,x);n=52
A=lambda w:np.c_[np.ones(len(w)),w]
p=s['panel_b'];B=np.linalg.lstsq(A(p['u']),p['v'],rcond=None)[0]
p=s['panel_c'];u=p['u']; F=lambda w,k:np.c_[A(w),w/(1+.84*abs(w)) if k=='v1' else np.sqrt(1+.959**2*w*w)]
Ws=[x,z,h,A(z)@B,A(x)@B]; Ws+= [F(w,k)@np.linalg.lstsq(F(u,k),p[k],rcond=None)[0] for w in Ws[:3] for k in ['v1','v2']]
R=np.random.default_rng(2); folds=[(np.arange(n),np.arange(n,72))]+[(p[:n],p[n:]) for p in [R.permutation(72) for i in range(7)]]
def blocks(tr,mode):
 K=[]
 def add(f):
  if f.ndim==1:f=f[:,None]
  q=f@f.T; scale=np.sqrt(np.mean(q[np.ix_(tr,tr)]**2)); K.append(q/max(scale,1e-12))
 def gram(q):K.append(q / np.sqrt(np.mean(q[np.ix_(tr,tr)]**2)))
 def pol(w,d):return np.stack([np.prod(w[:,ids],1) for ids in comb(range(4),d)],1)
 for w in ([x] if mode in ['iso','ard','bare'] else Ws[:5] if mode in ['full','sparse'] else Ws):
  v=w-w[tr].mean(0); G=v@v.T; d=np.maximum(G.diagonal()[:,None]+G.diagonal()[None,:]-2*G,0)
  if mode!='bare':
   for l in [.65,1.15,2.2]:gram(np.exp(-d/(2*l*l)))
  for j in range(2,6):
   f=pol(w,j)
   add(f)
   if (mode in ['ard','sparse'] and j<=3):
    for k in f.T:add(k)
  if mode not in ['full','sparse','all']:continue
  for j in [.22,1.25]:
   for f in [np.tanh(j*w),np.sin(j*w),w/(1+j*abs(w)),w/(1+j*w*w),np.sqrt(1+j*w*w)-1]:
    add(f);add(np.einsum('ni,nj->nij',f,w).reshape(len(w),-1))
   i,k=np.triu_indices(4,1); dd=w[:,i]-w[:,k]; ss=w[:,i]+w[:,k]
   for q in [dd,ss]:
    add(np.tanh(j*q));add(np.sin(j*q));add(np.einsum('ni,nj->nij',q/(1+j*q*q),w).reshape(len(w),-1))
  for j in [.22,1.25]:
   f=pol(w,2); add((f[:,:,None]/(1+j*w[:,None,:]**2)).reshape(len(w),-1))
 return np.array(K)
def predict(K,tr,te,y,reg):
 C=np.concatenate((K[:,tr][:,:,tr],np.eye(len(tr))[None]));D=K[:,te][:,:,tr]; base=2+2*x@x.T; bt=base[np.ix_(tr,tr)]; be=base[np.ix_(te,tr)]; r=y[tr]; eye=np.eye(len(tr)); p=np.full(len(C),-7.);p[-1]=-7.; CC=C.reshape(len(C),-1).copy()
 def opt(t):
  w=np.exp(t); c=cho_factor(bt+(w@CC).reshape(bt.shape),check_finite=False); a=cho_solve(c,r,check_finite=False); V=cho_solve(c,eye,check_finite=False)-np.outer(a,a)
  return .5*r@a+np.log(c[0].diagonal()).sum()+reg*np.sum(w[:-1]**.35), .5*w*(CC@V.ravel())+np.r_[.35*reg*w[:-1]**.35,0]
 fit=minimize(opt,p,jac=True,method='L-BFGS-B',bounds=[(-16,1)]*(len(C)-1)+[(-13,-2)],options={'maxiter':120,'ftol':1e-8}); w=np.exp(fit.x); c=cho_factor(bt+(w@CC).reshape(bt.shape)); out=(be+np.einsum('i,ijk->jk',w[:-1],D))@cho_solve(c,r);return out,np.sort(w[:-1])[-6:],w[-1]
if __name__=='__main__':
 out={}
 for mode in ['arch','iso','ard','bare','full','sparse','all']:
  for reg in ([0.,1.,3.] if mode not in ['arch','bare'] else [0.]):
   errs=[];P=[]
   for tr,te in folds:
    if mode=='arch':
     d={**s,'train':{**s['train'],'x':x[tr],'y':y[tr],'traj':T[tr]},'test':{'x':x[te]}}; a,b,_=old._components(d); pu=.65*a[:,0]+.35*b[:,0]
    else:
     pu,w,no=predict(blocks(tr,mode),tr,te,y,reg)
    errs.append(np.mean(abs(pu-y[te])));P.append(pu)
   out[mode,reg]=np.array(P); print(mode,reg,np.round(errs,4),np.mean(errs),flush=True)
 Y=np.array([y[te] for tr,te in folds]); base=out['arch',0.];
 for key,P in out.items():
  print('blend',key,np.round(np.mean(abs(.5*(base+P)-Y),1),4))
 np.savez('/tmp/gpstudy.npz',**{str(k):v for k,v in out.items()})
