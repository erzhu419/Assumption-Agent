import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import solve
np.set_printoptions(precision=5,suppress=True,linewidth=145)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; x=tr['x']; y=tr['y']; n=len(y)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def mon(x,d,rep=True):return np.array([np.prod(x[:,j],axis=1) for j in (itertools.combinations_with_replacement if rep else itertools.combinations)(range(4),d)]).T
def fit(F,V):
 C=lin(F,V);E=F@C-V; h=np.sum(F*np.linalg.pinv(F).T,axis=1);return np.sqrt(np.mean(E**2)),np.mean(abs(E)/np.maximum(1-h[:,None],1e-8)),C
A=s['panel_a']; u=A['u']; v=A['v']; keep=abs(v).max(1)<4;u=u[keep];v=v[keep]
print('A inliers',len(u));print('A linear residual',v-one(u)@lin(one(u),v))
for name,fun in [('poly2',lambda u:np.c_[one(u),mon(u,2)]),('cube',lambda u:np.c_[one(u),mon(u,2),u**3]),('poly3',lambda u:np.c_[one(u),mon(u,2),mon(u,3)]),('pair',lambda u:np.c_[one(u),mon(u,2,False)]),('trig',lambda u:np.c_[one(u),np.sin(1.25*u),np.cos(1.25*u)]),('soft',lambda u:np.c_[one(u),u/(1+.22*abs(u)),u*u]),('rational',lambda u:np.c_[one(u),u/(1+.22*u*u),u*u/(1+.22*u*u)]),('linear and all pairs''',lambda u:np.c_[one(u),mon(u,2,False),np.tanh(u),np.tanh(u)**2])]:
 F=fun(u);e,cv,C=fit(F,v); print('A',name,e,cv);print(C if cv<.03 else '')
p=s['panel_c'];u=p['u'];v=p['v2'];
for name,fun in [('sin',lambda u,k:np.sin(k*u)),('cos',lambda u,k:np.cos(k*u)),('sincos',lambda u,k:np.c_[np.sin(k*u),np.cos(k*u)]),('tanh',lambda u,k:np.tanh(k*u)),('expquad',lambda u,k:np.exp(-k*u*u)),('rational',lambda u,k:1/(1+k*u*u)),('exp',lambda u,k:np.exp(k*u)),('soft',lambda u,k:u/(1+k*abs(u)))]:
 for base in [lambda u:np.ones((len(u),1)),one,lambda u:np.c_[one(u),u*u],lambda u:np.c_[one(u),mon(u,2,False)]]:
  def feat(k):return np.c_[base(u),fun(u,k[0])];
  def res(k):
   F=feat(k);return (F@lin(F,v)-v).ravel()
  o=min([least_squares(res,[k],bounds=(.02,7)) for k in [.3,1.25,2.7,4.2]],key=lambda o:o.cost);e,cv,C=fit(feat(o.x),v); print('C2',name,base(u).shape[1],o.x,e,cv)
  if cv<.001: print(C)
def tm(r):return -np.expm1(-t[:,None]*r)
def res(r):
 F=tm(r);return (F@lin(F,T.T)-T.T).ravel()
r=least_squares(res,[.38,1.8,5.5],bounds=(.03,15)).x; H=lin(tm(r),T.T).T; print('time rates',r,'H ranges/std',H.min(0),H.max(0),H.std(0));print('y vs H',fit(one(H),y[:,None])[:2],fit(np.c_[one(H),H[:,[0,0,0,1,1,2]]*H[:,[0,1,2,1,2,2]]],y[:,None])[:2]);print(np.c_[x,y,H][:30])
