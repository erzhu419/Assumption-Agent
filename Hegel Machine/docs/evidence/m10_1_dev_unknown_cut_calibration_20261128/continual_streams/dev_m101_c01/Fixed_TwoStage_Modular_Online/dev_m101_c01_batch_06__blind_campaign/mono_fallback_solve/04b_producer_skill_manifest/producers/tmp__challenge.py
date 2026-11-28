from probe import *
from composed import spaces,folds
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import importlib.util
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/dev_m101_c01_batch_02_solver.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def predict(x,q,y,groups):
 n=len(x); xs=np.r_[x,q]; D=design(xs,2); D[:,5:]*=.5
 K=[D[:,:1]@D[:,:1].T,D[:,1:5]@D[:,1:5].T,D[:,5:]@D[:,5:].T]
 for a in groups:
  a=np.asarray(a); a=(a-a[:n].mean(0))/np.maximum(.02,a[:n].std(0)); K.append(a@a.T/a.shape[1]);
 K=np.asarray(K); train=K[:,:n,:n]; cross=K[:,n:,:n]; eye=np.eye(n); Y=np.asarray(y); center=Y.mean(0); scale=np.maximum(1e-8,Y.std(0)); Y=(Y-center)/scale
 if Y.ndim==1:Y=Y[:,None]
 def obj(p,ret=False):
  w=np.exp(p); M=np.einsum('i,ijk->jk',w[:-1],train)+w[-1]*eye; fac=cho_factor(M,check_finite=False); c=cho_solve(fac,Y,check_finite=False)
  if ret:return (np.einsum('i,ijk->jk',w[:-1],cross)@c).squeeze()*scale+center
  G=Y.shape[1]*cho_solve(fac,eye,check_finite=False)-c@c.T
  v=.5*np.sum(Y*c)+Y.shape[1]*np.log(fac[0].diagonal()).sum(); g=np.r_[np.einsum('ijk,jk->i',train,G),np.trace(G)]*w*.5
  return v,g
 p=minimize(obj,np.r_[np.full(3,-1.),np.full(len(K)-3,-5.),-8.],jac=True,method='L-BFGS-B',bounds=[(-13,2)]*len(K)+[(-12,-2)],options={'maxiter':100}).x
 return obj(p,True)
def group(q,kind):
 out=[]
 for u in q:
  if kind=='smooth':
   out += [np.c_[np.sin(1.6*u),np.cos(1.6*u)],np.c_[np.sin(2.8*u),np.cos(2.8*u)],poly(u,3),poly(u,4)]
  if kind=='soft':
   out += [u/(1+.83*abs(u)),(u[:,:,None]*(u/(1+.83*abs(u)))[:,None,:]).reshape(len(u),-1),poly(u/(1+.83*abs(u)),2)]
  if kind=='mixed':
   v=np.c_[u/(1+.83*abs(u)),np.cos(1.29*u),np.sin(1.6*u)];out += [v,(u[:,:,None]*v[:,None,:]).reshape(len(u),-1),poly(u,3),poly(u,4)]
 return out
if __name__=='__main__':
 x,y,z=[S['train'][k] for k in ('x','y','traj')];t=S['eval_times']; sp=spaces(S,x); ss=list(folds(x)); print('telemetry',S['telemetry_vec'],'summary',S['train']['summary'])
 for k in ['old','raw','a','ai','b','ba','bai','ab','all']:
  for kind in (['archive'] if k=='old' else ['smooth','soft','mixed']):
   all=[]
   for i,j in ss:
    if k=='old':
     yp,zz=old._components(x[i],y[i],z[i],x[j]); yh=yp@np.array([.25,.30,.30,.15]);zh=np.average(zz,axis=0,weights=[.4,.2,.4])
    else:
     idx=np.r_[i,j]; groups=group([sp[s][idx] for s in ({'raw':['x'],'all':['ai','ba','ab']}.get(k,[k]))],kind); yh=predict(x[i],x[j],y[i],groups); _,_,basis=np.linalg.svd(z[i],full_matrices=False); basis=basis[:2]; zh=np.column_stack([predict(x[i],x[j],amp,groups) for amp in (z[i]@basis.T).T])@basis
    all.append([np.mean(abs(yh-y[j])),np.mean(abs(zh-z[j]))]);
   print(k,kind,np.round(np.array(all).T,5),flush=True)
