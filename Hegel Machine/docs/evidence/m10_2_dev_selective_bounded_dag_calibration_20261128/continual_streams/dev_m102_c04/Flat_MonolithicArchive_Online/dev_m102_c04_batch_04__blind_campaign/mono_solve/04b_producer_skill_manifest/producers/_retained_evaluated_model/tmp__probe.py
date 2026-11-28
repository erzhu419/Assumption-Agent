import numpy as np,pickle,sys
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_m102_c04_batch_00_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; ls=np.linalg.lstsq
np.set_printoptions(precision=5,suppress=True,linewidth=150)
a,M=m._calibration(s['panel_a']); print('times',t,'A',a,M); print('a residual',np.sort(np.linalg.norm(s['panel_a']['u']@M+a-s['panel_a']['v'],axis=1))); p=x@M+a; q=(x-a)@np.linalg.inv(M)
def test(name,F):
 F=np.c_[np.ones(len(x)), F]; Y=np.c_[y,T]; pred=F@ls(F[:52],Y[:52],rcond=1e-10)[0]; e=np.abs(Y-pred); print(name,F.shape[1],np.round([e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean()],5))
def poly(x,d):return np.array([np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)]).T
for d in range(1,4):test('raw poly'+str(d),poly(x,d))
for n,z in [('x',x),('p',p),('q',q)]:
 for k in [.16,.22,.5,.8,1.25]:
  R=np.c_[z,z/(1+k*abs(z)),z*z/(1+k*z*z)]; test((n,k),R); test((n,k,'+poly'),np.c_[R,poly(x,2)[:,4:]])
f1,f2=m._panel_operators(s['panel_c']);
for name,z in [('x',x),('p',p),('q',q)]:
 u,v=f1(z),f2(z); test((name,'C'),np.c_[z,u,v]); test((name,'C+prod'),np.c_[z,u,v,np.einsum('ni,nj->nij',u,v).reshape(-1,16)]);
for n,z in [('b',s['panel_b']['v']),('c1',s['panel_c']['v1']),('c2',s['panel_c']['v2'])]:
 u=s['panel_'+n[0]]['u']; print('PANEL',n)
 for name,fn in [('linear',lambda u,k:u),('oddabs',lambda u,k:u/(1+k*abs(u))),('even',lambda u,k:u*u/(1+k*u*u)),('tanh',lambda u,k:np.tanh(k*u)),('sin',lambda u,k:np.sin(k*u)),('oddsq',lambda u,k:u/(1+k*u*u))]:
  def X(k):return np.c_[np.ones(len(u)),u,fn(u,k[0])]
  def fun(k):
   F=X(k);return (F@ls(F,z,rcond=1e-10)[0]-z).ravel()
  opt=least_squares(fun,[.5],bounds=(.01,4)); print(name,opt.x,np.std(opt.fun))
for form in [0,1,2,3]:
 def B(r):
  if form==0:return -np.expm1(-t[:,None]*r)
  if form==1:return np.c_[-np.expm1(-t*r[0]),t]
  if form==2:return np.c_[-np.expm1(-t*r[0]),t*np.exp(-t*r[1])]
  if form==3:return np.c_[-np.expm1(-t*r[0]),np.sin(t*r[1])]
 def fun(r):
  F=B(r);return (F@ls(F,T[:52].T,rcond=1e-10)[0]-T[:52].T).ravel()
 opt=least_squares(fun,[1.8,.43],bounds=([.01,.001],[25,20])); F=B(opt.x); print('time',form,opt.x,np.std(opt.fun),np.mean(abs(F@ls(F,T[52:].T,rcond=1e-10)[0]-T[52:].T)))
