import pickle,sys,importlib.util
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); t=s['eval_times']; T=tr['traj']; y=tr['y']; x=tr['x']; a=np.arange(52); b=np.arange(52,72)
def load(i):
 p=f'/tmp/hegel_continual_archive/conf_m101_b03_batch_0{i}_solver.py'; sp=importlib.util.spec_from_file_location('arc',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
m=load(1); c=m._calibrate(s['panel_a']); z=m._inverse(x,c); print('calibration',c.round(5)); X=np.c_[np.ones(n),z]; P=s['panel_b']; B=X@np.linalg.lstsq(np.c_[np.ones(len(P['u'])),P['u']],P['v'],rcond=None)[0];
from scipy.special import erf
from scipy.optimize import minimize_scalar
from itertools import combinations
P=s['panel_c'];u=P['u'];v=P['v1'];j=np.arange(len(u));fit=lambda F,V:np.linalg.lstsq(F,V,rcond=1e-10)[0]
def aff(q):return np.c_[np.ones(len(q)),q]
def fns():
 return {'tanh':np.tanh,'atan':np.arctan,'erf':erf,'root':lambda x:x/np.sqrt(1+x*x),'rat':lambda x:x/(1+x*x),'sin':np.sin,'asinh':np.arcsinh,'soft':lambda x:x/(1+abs(x))}
results=[]
for nm,fn in fns().items():
 for typ,start in [('shared',[1.5]),('shift',[1.5,0.]),('linear',[1.5,0.,0.]),('separate',[1.5]*4),('sep_shift',[1.5]*4+[0.]*4)]:
  def basis(x,k,fn=fn,typ=typ):
   if typ=='shared': g=fn(k[0]*x)
   elif typ=='shift': g=fn(k[0]*x+k[1])
   elif typ=='linear':g=fn(k[0]*x+k[1])+k[2]*x
   elif typ=='separate':g=fn(k*x)
   else:g=fn(k[:4]*x+k[4:])
   return aff(g)
  def est(ids):
   def res(k):
    a=basis(u[ids],k);return (a@fit(a,v[ids])-v[ids]).ravel()
   ns=4 if typ.startswith('sep') else 1
   return least_squares(res,start,bounds=([.15]*ns+[-4]*(len(start)-ns),[7.]*ns+[4]*(len(start)-ns)),max_nfev=100).x
  ks=est(j);err=[]
  for i in range(4):
   ids=j[j%4!=i];out=j[j%4==i];k=est(ids);err.extend(basis(u[out],k)@fit(basis(u[ids],k),v[ids])-v[out])
  results.append((np.sqrt(np.mean(np.square(err))),nm,typ,ks,basis))
results.sort(key=lambda q:q[0]); print('panel1',[(e,nm,typ,k.round(5)) for e,nm,typ,k,_ in results[:18]])
_,_,_,k,basis=results[0]; p=basis(z,k)@fit(basis(u,k),v); q=aff(np.cos(2.4*z))@fit(aff(np.cos(2.4*u)),P['v2'])
def outer(u,v):return (u[:,:,None]*v[:,None,:]).reshape(len(u),-1)
from scipy.linalg import solve
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
X=np.c_[z,p,q,outer(B,p),outer(p,q)]; X0=X[a].mean(0);Xs=X[a].std(0);F=(X-X0)/Xs;Y=np.c_[y,T]; mu=Y[a].mean(0);print('ridge errors')
for l in [1e-6,.0001,.001,.01,.03,.1]:
 w=solve(F[a].T@F[a]+l*np.eye(F.shape[1]),F[a].T@(Y[a]-mu),assume_a='pos'); E=F[b]@w+mu-Y[b];print(l,np.mean(abs(E[:,0])),np.mean(abs(E[:,1:])),np.mean(abs(E[:,-1])))
print('sparse fits')
for tar in [0,1,3,8]:
 for l in [.00002,.0001,.0005,.002,.01]:
  m=Lasso(alpha=l,max_iter=25000,tol=1e-7).fit(F[a],Y[a,tar]);bcoef=m.coef_/Xs;ids=np.argsort(abs(bcoef))[-8:][::-1];print(tar,l,round(np.mean(abs(m.predict(F[b])-Y[b,tar])),5),list(zip(ids,bcoef[ids].round(3))))
np.savez('/tmp/refined03.npz',z=z,B=B,p=p,q=q,y=y,T=T,t=t)
