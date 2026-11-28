import pickle
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda x,y: np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x):return np.column_stack([np.ones(len(x)),x])
def quad(x):return np.column_stack([aff(x)]+[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)])
u,v=(s['panel_a'][k] for k in ('u','v')); d=aff(u); w=ls(d,v)
for _ in range(12):
 e=np.linalg.norm(d@w-v,axis=1); wt=1/np.maximum(e,1e-10); w=ls(d*wt[:,None]**.5,v*wt[:,None]**.5)
e=np.linalg.norm(d@w-v,axis=1); print('A residual norms',np.sort(e)); print('A forward',w)
x=s['train']['x']; print('raw X and inverse A ranges',np.stack([np.min(x,0),np.max(x,0),np.min((x-w[0])@np.linalg.inv(w[1:]),0),np.max((x-w[0])@np.linalg.inv(w[1:]),0)]))
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=(s[pn][k] for k in ('u',vn)); print(pn,vn,'ranges',v.min(0),v.max(0))
 for d in [aff(u),quad(u)]:
  res=lambda y: y-d@ls(d,y)
  print('design',d.shape[1], 'unit inverse links rms',[(n,round(np.mean(res(f(v))**2)**.5,7)) for n,f in [('id',lambda x:x),('asinh',np.arcsinh),('atanh',lambda x:np.arctanh(np.clip(x,-.999999,.999999))),('asin',lambda x:np.arcsin(np.clip(x,-1,1))),('tan',np.tan)]])
  for n,f in [('tanh',np.tanh),('sin',np.sin),('cubic',lambda x:x+x**3),('rational',lambda x:x/(1+x*x)**.5),('softsign',lambda x:x/(1+abs(x)))]:
   def fit(j):
    y=v[:,j]; a=max(abs(y))*1.2
    def pred(p):return p[0]+np.exp(p[1])*f(d@p[2:])
    p=np.r_[0,np.log(a),ls(d,y/a)]; r=least_squares(lambda p:pred(p)-y,p,max_nfev=350,bounds=(np.r_[[-5,-6],np.full(len(p)-2,-30)],np.r_[[5,4],np.full(len(p)-2,30)]));return np.linalg.norm(r.fun),r.x[:2]
   print(n,[(round(err,6),np.round(p,3).tolist()) for err,p in [fit(j) for j in range(4)]])
 if pn=='panel_b':
  print('B anchors u|v',np.round(np.column_stack([u,v]),4))
