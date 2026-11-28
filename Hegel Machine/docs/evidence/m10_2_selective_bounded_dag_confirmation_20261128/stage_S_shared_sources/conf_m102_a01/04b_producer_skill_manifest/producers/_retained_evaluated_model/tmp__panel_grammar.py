import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; C=s['panel_c']; u=C['u']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
def aug(z): return np.c_[np.ones(len(z)),z]
def fit(name,v,fn,start,bounds):
 def res(k):
  F=fn(u,*k); return (F@np.linalg.lstsq(F,v,rcond=None)[0]-v).ravel()
 R=least_squares(res,start,bounds=bounds,max_nfev=300); F=fn(u,*R.x); W=np.linalg.lstsq(F,v,rcond=None)[0]; h=np.sum(F*np.linalg.pinv(F).T,axis=1); e=v-F@W
 print(name,R.x,'rmse',np.std(e),'approx loo',np.std(e/(1-h[:,None]))); return R.x,W
v=C['v2']
for key,fun,k,b in [('rational',lambda z,b:z*z/(1+b*z*z),[.1],([0],[3])),('sqrt',lambda z,b:(np.sqrt(1+b*z*z)-1)/b,[.1],([.001],[3])),('cos',lambda z,b:1-np.cos(b*z),[.5],([.001],[3])),('tanh2',lambda z,b:np.tanh(b*z*z),[.1],([.001],[3]))]:
 fn=lambda z,b:np.c_[aug(z),fun(z,b)]; p,w=fit(key,v,fn,k,b); print('unmixed quad',w[5:]@np.linalg.inv(w[1:5]));
v=C['v1']; cross=lambda z:np.array([z[:,i]*z[:,j] for i,j in combinations(range(4),2)]).T
for key,fun in [('tanh',lambda z,b:np.tanh(b*z)),('sin',lambda z,b:np.sin(b*z)),('rational',lambda z,b:z/(1+b*z*z)),('atan',lambda z,b:np.arctan(b*z))]:
 for extra in range(4):
  fn=lambda z,b:np.c_[aug(fun(z,b)),*([] if extra==0 else [z] if extra==1 else [cross(z)] if extra==2 else [z,cross(z)])]; p,w=fit((key,extra),v,fn,[.8],([.001],[5]));
  if key=='tanh' and extra==2: print('unmixed cross',w[5:]@np.linalg.inv(w[1:5]));
 fn=lambda z,b:np.c_[aug(fun(z,b)),z*z,cross(z)]; p,w=fit((key,'all quad'),v,fn,[.8],([.001],[5]));
for b in [.5,1,1.5,2]:
 F=aug(np.tanh(u*b)); w=np.linalg.lstsq(F,v,rcond=None)[0]; print('tanh sv',b,np.linalg.svd(w[1:],compute_uv=False));
