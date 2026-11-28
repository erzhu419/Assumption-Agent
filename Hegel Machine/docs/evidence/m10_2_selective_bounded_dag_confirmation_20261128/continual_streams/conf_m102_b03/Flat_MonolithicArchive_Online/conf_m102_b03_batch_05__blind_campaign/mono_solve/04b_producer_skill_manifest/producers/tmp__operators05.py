import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from itertools import combinations_with_replacement
from calibrate05 import _calibrate
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(d,y):return np.linalg.lstsq(d,y,rcond=None)[0]
def one(x):return np.column_stack([np.ones(len(x)),x])
def poly(x,deg):return np.column_stack([np.prod(x[:,k],axis=1) for l in range(1,deg+1) for k in combinations_with_replacement(range(4),l)])
def cyc(x):return x*np.roll(x,-1,axis=1)
fams={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'softsign':lambda x:x/(1+abs(x)),'swish':lambda x:x*expit(x),'softplus':lambda x:np.logaddexp(0,x),'gauss':lambda x:np.exp(-x*x),'rational':lambda x:1/(1+x*x),'odd_rat':lambda x:x/(1+x*x),'asinh':np.arcsinh,'atan':np.arctan,'relu_sq':lambda x:np.maximum(0,x)**2}
bases={'0':lambda x:np.zeros((len(x),0)),'lin':lambda x:x,'cyc':cyc,'sq':lambda x:x*x,'lincyc':lambda x:np.c_[x,cyc(x)]}
inputs={'u':lambda x:x,'cyc':cyc,'sum':lambda x:x+np.roll(x,-1,axis=1),'diff':lambda x:x-np.roll(x,-1,axis=1)}
p=s['panel_c']; u=p['u'];
for key in ('v1','v2'):
 v=p[key]; ans=[]
 for bn,b in bases.items():
  for xn,tr in inputs.items():
   w=tr(u)
   for fn,f in fams.items():
    def des(q):return one(np.c_[b(u),f(q*w)])
    def loss(q):
     d=des(q);return np.mean((d@ls(d,v)-v)**2)
    r=minimize_scalar(loss,bounds=(.1,5),method='bounded');ans.append((np.sqrt(r.fun),bn,xn,fn,r.x))
 for bn,b in {**bases,'quad':lambda x:poly(x,2)}.items():
  d=one(b(u));ans.append((np.sqrt(np.mean((d@ls(d,v)-v)**2)),bn,'','none',0))
 print(key,sorted(ans)[:18]);
 for _,bn,xn,fn,q in sorted(ans)[:3]:
  if fn=='none':continue
  d=one(np.c_[bases[bn](u),fams[fn](q*inputs[xn](u))]);print(np.round(ls(d,v),4))
x=np.vstack((s['train']['x'],s['test']['x'])); z=_calibrate(s['panel_a'],x); print('calibrated range',z.min(0),z.max(0)); Y=np.c_[s['train']['y'],s['train']['traj']];
for k,w in [('z',z),('b',one(z)@ls(one(s['panel_b']['u']),s['panel_b']['v']))]:
 for degree in [1,2,3]:
  d=one(poly(w,degree));err=abs(d[52:72]@ls(d[:52],Y[:52])-Y[52:]);print(k,degree,err[:,0].mean(),err[:,1:].mean())
