import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];Y=s['train']['y'];Z=s['train']['traj'];T=np.linalg.svd(Z,full_matrices=False)[2][:2]; targets=np.c_[Y,Z@T.T]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
def panel(u,v,fun):
 f=fun(u);keep=np.ones(len(u),bool)
 for j in range(5):
  b=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0];e=np.linalg.norm(f@b-v,axis=1);keep=e<3
 return b,e
u,v=(s['panel_a'][k] for k in ('u','v')); f,e=panel(u,v,lambda x:poly(x,1));print('a fit',np.round(f,3));good=e<3
for name,fun in [('linear',lambda x:poly(x,1)),('sq',lambda x:poly(x,2)),('cube',lambda x:poly(x,3)),('axis-sq',lambda x:np.c_[poly(x,1),x*x]),('axis-cube',lambda x:np.c_[poly(x,1),x*x,x**3]),('sine',lambda x:np.c_[poly(x,1),np.sin(2.15*x),np.cos(2.15*x)]),('tanh',lambda x:np.c_[poly(x,1),np.tanh(2.15*x)])]:
 a=fun(u)[good];b=v[good];p=np.linalg.pinv(a);r=a@(p@b)-b;loo=r/(1-np.sum(a*p.T,axis=1))[:,None]; print('A',name,np.round(np.mean(r*r,axis=0)**.5,4),np.round(np.mean(loo*loo,axis=0)**.5,4))
c=s['panel_c'];b=s['panel_b'];matrix=np.linalg.lstsq(poly(c['u'],1),c['v1'],rcond=None)[0]; print('map sv',np.linalg.svd(matrix[1:],compute_uv=False));print('corr a err',np.round(np.corrcoef(np.c_[u,v-poly(u,1)@f][good].T),2));print('b dist corr',np.corrcoef(*[np.linalg.norm(b[k][:,None]-b[k][None,:],axis=-1)[np.triu_indices(len(b[k]),1)] for k in ('u','v')])[0,1]);
q=(x-f[0])@np.linalg.inv(f[1:]); qdiag=(x-f[0])/f[1:].diagonal(); qtrim=(x-np.array([.0715,-.0926,-.0774,.0427]))@np.linalg.inv(np.array([[.84,-.029,-.039,.274],[.262,1.094,-.033,-.006],[-.007,.309,1.203,-.036],[.015,-.011,.218,1.185]]));
def calc(A,inds=np.arange(52),test=np.arange(52,72)):
 a=A[inds]; B=A[test];U,d,V=np.linalg.svd(a,full_matrices=False); m=U.T@targets[inds]; errs=[]
 for l in [1e-8,1e-5,.0001,.001,.003,.01,.03,.1,.3,1,3]:
  p=(B@V.T)@((d/(d*d+l))[:,None]*m);errs.append([np.mean(abs(p[:,0]-Y[test])),np.mean(abs((p[:,1:]-targets[test,1:])@T))])
 return np.array(errs).min(0)
res=[]
for name,z in [('raw',x),('diag',qdiag),('full',q),('trim',qtrim)]:
 for mapname,w in [('axis',z),('v1',poly(z,1)@matrix),('v2',poly(z,1)@np.linalg.lstsq(poly(c['u'],1),c['v2'],rcond=None)[0])]:
  for scale in [0,.03,.09,.22,.5,1]:
   h=w/np.sqrt(1+scale*w*w)
   for d in [2,3,4,5]:
    p=poly(h,d);res.append((calc(p),name,mapname,scale,d))
   for freq in [.5,1,1.25,1.5,2.15,2.5,3]:
    p=np.c_[poly(h,2),np.sin(freq*h),np.cos(freq*h)];res.append((calc(p),name,mapname,scale,('four',freq)))
for j in [0,1]:
 print('best',j); [print(r) for r in sorted(res,key=lambda a:a[0][j])[:12]]
