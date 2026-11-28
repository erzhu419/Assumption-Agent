import sys, pickle, itertools
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
import numpy as np
import node_00 as m, node_01 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
z,b,c1,c2=m._features(s); a=s['panel_a']; c=s['panel_c']; t=s['train']; print('decoded ranges',z.min(0),z.max(0))
print('A median inverse error',np.median(np.linalg.norm(m._decode(a,a['v'])-a['u'],axis=1)))
u,v=c['u'],c['v2']
from scipy.optimize import minimize_scalar
for name,fn in [('cos',lambda u,p:(1-np.cos(p*u))/p**2),('rational',lambda u,p:u*u/(1+p*u*u)),('exp',lambda u,p:-np.expm1(-p*u*u)/p),('log',lambda u,p:np.log1p(p*u*u)/p),('sqrt',lambda u,p:(np.sqrt(1+p*u*u)-1)/p)]:
 def obj(p,report=False):
  d=m._design(u,fn(u,p)); e=v-d@m._fit(d,v); q=np.linalg.qr(d)[0]; return np.sqrt(np.mean((e/(1-(q*q).sum(1)[:,None]))**2)) if report else np.mean(e*e)
 res=minimize_scalar(obj,bounds=(.001,2.5),method='bounded'); print('v2',name,res.x,res.fun**.5,obj(res.x,True))
y,st=t['y'],t['traj']; n=len(y); inds=np.arange(n); splits=[np.arange(52,n)]+list(np.array_split(np.random.default_rng(52).permutation(n),5))
def score(h):
 h=f._readout_basis(h[:n]); gram=h@h.T; out=[]
 for te in splits:
  tr=np.setdiff1d(inds,te); g=gram[np.ix_(tr,tr)]; cross=gram[np.ix_(te,tr)]; eye=np.eye(len(tr)); dy=cross@np.linalg.solve(g+.0003*eye,y[tr])-y[te]; ds=cross@np.linalg.solve(g+.001*eye,st[tr])-st[te]; out.append([np.mean(abs(dy)),np.mean(abs(ds))])
 return np.round(out[0],5),np.round(np.max(out,axis=0),5),np.round(np.mean(out[1:],axis=0),5)
for k in range(1,5):
 for group in itertools.combinations(zip(['z','b','c1','c2'],[z,b,c1,c2]),k):
  nm,x=zip(*group); print(''.join(nm),score(np.column_stack(x)))
