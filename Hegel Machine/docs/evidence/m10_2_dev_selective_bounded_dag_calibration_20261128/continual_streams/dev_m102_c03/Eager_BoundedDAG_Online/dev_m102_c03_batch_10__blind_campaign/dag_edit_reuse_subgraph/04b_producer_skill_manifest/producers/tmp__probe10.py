import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
for panel,key in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 U=S[panel]['u']; V=S[panel][key]; n=len(U); print('\n',panel,key)
 def assess(f, desc):
  C=np.linalg.lstsq(f,V,rcond=None)[0]; R=V-f@C; print(desc,np.sqrt(np.mean(R**2,axis=0)),np.sort(np.linalg.norm(R,axis=1))[-7:]); return C,R
 F=np.c_[np.ones(n),U]; C,R=assess(F,'affine'); print(C)
 names=['square','cubic cyclic','rational pair cyclic','tanh','cos','exp','sine']
 for name,k in zip(names,[lambda x:x*x,lambda x:x*np.roll(x,-1,axis=1)*np.roll(x,-2,axis=1),lambda x:x*np.roll(x,-1,axis=1)/(1+x*x+np.roll(x,-1,axis=1)**2),np.tanh,lambda x:np.cos(1.25*x),lambda x:np.expm1(-.3*x*x),lambda x:np.sin(1.25*x)]):
  C,R=assess(np.c_[F,k(U)],name)
  if np.std(R)<.025: print('extra',C[5:])
 if panel=='panel_a':
  for j,(x,y) in enumerate(zip(U.T,V.T)):
   for fn in [lambda x,k:np.cos(k*x)-1,lambda x,k:np.expm1(-k*x*x),lambda x,k:np.tanh(k*x),lambda x,k:x*x/(1+k*x*x)]:
    sol=least_squares(lambda p:p[0]+p[1]*x+p[2]*fn(x,p[3])-y,[0,1,.2,.8],loss='soft_l1',f_scale=.01); print(j,sol.x,np.quantile(abs(sol.fun),[.5,.8,.95,1]))
T=S['eval_times']; Y=S['train']['traj']; print('sv time',np.linalg.svd(Y,compute_uv=False));
for r in [[.39,1.85],[.3,1,2.5],[.15,.4,1.4,2.5]]:
 def fun(r):
  Q=-np.expm1(-T[:,None]*r); return (Q@np.linalg.lstsq(Q,Y.T,rcond=None)[0]-Y.T).ravel()
 fit=least_squares(fun,r,bounds=(.00001,10)); print('rates',fit.x,'rms',np.std(fit.fun),'time rmse',np.sqrt(np.mean(fit.fun.reshape(Y.T.shape)**2,axis=1)))
