import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _nodes,_features,solve_batch
import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];p=s['train'];x=p['x']; y=p['y'];t=p['traj'];Y=np.c_[y,t]
def ls(g,v):return np.linalg.lstsq(g,v,rcond=None)[0]
a,b,c=_nodes(s,x); G=_features(a,c)
print('trajectory singular values',np.linalg.svd(t,compute_uv=False))
print('coeff full diagnostic only: intercept, 8 linear nodes, then quadratic upper triangle')
print(np.round(ls(np.c_[np.ones(len(x)),G],Y)[:,[0,1,4,8]],3))
for label,g in [('linear',np.c_[a,c]),('linear+b',np.c_[a,b,c]),('linear+diag',np.c_[a,c,a*c]),('linear+xprod',np.c_[a,c,(a[:,:,None]*c[:,None]).reshape(len(a),-1)]),('quadratic',G)]:
 print(label,end=' ')
 for ix,iv in [(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]:
  z=np.c_[np.ones(len(g)),g];e=abs(z[iv]@ls(z[ix],Y[ix])-Y[iv]);print(np.round([e[:,0].mean(),e[:,1:].mean()],5),end=' ')
 print()
pa=s['panel_a'];u,v=pa['u'],pa['v'];z=np.c_[np.ones(len(u)),u];k=ls(z,v)
for _ in range(12):
 ids=np.argsort(np.linalg.norm(v-z@k,axis=1))[:38];k=ls(z[ids],v[ids]);
print('A full estimate',np.round(k,5),'res quantiles',np.quantile(np.linalg.norm(v-z@k,axis=1),[.4,.6,.8,.9]))
for k,v in [('v1',s['panel_c']['v1']),('v2',s['panel_c']['v2'])]:
 u=s['panel_c']['u']; f=(lambda r:-np.sign(u)*np.expm1(-r*abs(u))) if k=='v1' else (lambda r:np.logaddexp(0,r*u))
 def obj(r):
  z=np.c_[np.ones(len(u)),f(r)];return np.mean((z@ls(z,v)-v)**2)
 opt=minimize_scalar(obj,bounds=(.2,5),method='bounded'); print(k,opt.x,np.sqrt(opt.fun),np.sqrt(obj(1.8 if k=='v1' else 2.2)))
