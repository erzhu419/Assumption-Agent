import numpy as np,pickle,sys
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a04_batch_00_solver as arc
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
one=arc.one
lin=arc.lin
def poly(u,d):
 return np.column_stack([np.prod(u[:,j],axis=1) for i in range(d+1) for j in combinations_with_replacement(range(u.shape[1]),i)])
def rfit(f,v):
 r=f@lin(f,v)-v
 for i in range(12):
  w=1/np.maximum(np.linalg.norm(r,axis=1),.001)
  r=f@lin(f*w[:,None]**.5,v*w[:,None]**.5)-v
 return r
p=s['panel_a'];u,v=p['u'],p['v']; a,b=arc._align(p)
print('archived align',a,b);print('archived residual row norm',np.round(np.linalg.norm(u*b+a-v,axis=1),4))
for k in range(1,4):
 f=poly(u,k);r=rfit(f,v); print('panel A degree',k,'quantile',np.quantile(np.linalg.norm(r,axis=1),[0,.25,.5,.75,.85,.9,1]),'coeff',np.round(lin(f[np.linalg.norm(r,axis=1)<np.quantile(np.linalg.norm(r,axis=1),.75)],v[np.linalg.norm(r,axis=1)<np.quantile(np.linalg.norm(r,axis=1),.75)]),4) if k==1 else '')
for name,keys in [('panel_b',['v']),('panel_c',['v1','v2'])]:
 p=s[name]; u=p['u']
 for key in keys:
  v=p[key];w=u*u
  print(name,key,'loo rmse and in sample')
  D={'affine':one(u),'p2':poly(u,2),'sq':np.c_[one(u),w],'sqrat':np.c_[one(u),w,w/(1+.25*w)],'tanh':one(np.tanh(u)), 'soft':one(u/(1+.85*abs(u))-.01*u), 'p2c':np.c_[poly(u,2),u**3]}
  for kind,f in D.items():
   r=f@lin(f,v)-v; h=np.diag(f@np.linalg.pinv(f)); print(kind,np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-h[:,None]))**2)))
  def fit(feat,x0,lb,ub):
   def res(q):
    f=feat(q);return (f@lin(f,v)-v).ravel()
   z=least_squares(res,x0,bounds=(lb,ub),gtol=1e-12);return z.x,np.std(z.fun)
  for tag,feat,x0,lb,ub in [('soft',lambda q:one(u/(1+q[0]*abs(u))+q[1]*u),[.8,0],[.001,-5],[20,5]),('tanh',lambda q:one(np.tanh(q[0]*u)+q[1]*u),[1,0],[.001,-5],[20,5]),('sin',lambda q:one(np.sin(q[0]*u)+q[1]*u),[1,0],[.001,-5],[20,5]),('rat',lambda q:one(u/(1+q[0]*w)+q[1]*u),[.5,0],[.001,-5],[20,5]),('curvature',lambda q:np.c_[one(u),w,w/(1+q[0]*w)],[.3],[.0001],[20])]: print(tag,fit(feat,x0,lb,ub))
