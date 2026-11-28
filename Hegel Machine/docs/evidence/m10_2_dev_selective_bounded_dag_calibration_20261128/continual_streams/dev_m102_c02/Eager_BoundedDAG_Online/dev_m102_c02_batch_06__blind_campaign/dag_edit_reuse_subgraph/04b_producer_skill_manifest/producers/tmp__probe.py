import sys,pickle,numpy as np
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00 as n
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=n.run_node({'__input__':s}); c=p['c']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; a=s['panel_a']; b=s['panel_b']; pc=s['panel_c']; aa=n._fit_a(**a); bs,bb=n._fit_b(**b); u=n._invert_a(x,aa); Y=np.c_[y,t];
print('A row error',np.round(np.sort(np.linalg.norm((n._a_design(a['u'])@aa.T[...,None])[...,0].T-a['v'],axis=1)),5)); print('B scale',bs,'err',np.std(n._b_design(b['u'],bs)@bb-b['v'],axis=0)); print('sv C, trajectory',np.linalg.svd(c,compute_uv=False),np.linalg.svd(t,compute_uv=False)); print('ranges u',u.min(0),u.max(0)); print('panel-c ranges',[(k,v.min(0),v.max(0)) for k,v in pc.items()]);
def poly(z,d):
 return np.stack([np.prod(z[:,j],axis=1) for i in range(d+1) for j in combinations_with_replacement(range(z.shape[1]),i)],axis=1)
for name,z in [('pc raw u',pc['u']),('pc B(u)',n._b_design(pc['u'],bs)@bb),('pc Ainverse(u)',n._invert_a(pc['u'],aa)),('pc BAinverse(u)',n._b_design(n._invert_a(pc['u'],aa),bs)@bb)]:
 for target in ['v1','v2']:
  for degree in [1,2]:
   f=poly(z,degree); v=pc[target]; coef=np.linalg.lstsq(f,v,rcond=None)[0]; e=abs(f@coef-v); loo=e/np.maximum(1e-10,1-np.sum(f*np.linalg.pinv(f).T,1))[:,None]; print(name,target,degree,'fit/LOO',e.mean(),loo.mean());
   if name=='pc raw u' and degree==1: print(np.round(coef,4))
for z in [u,c,x]:
 for k in range(4):
  o=np.argsort(z[:,k]); print('rank pattern',k, np.round(np.c_[z[o,k],y[o]][::5],2).tolist())
print('large errors c linear'); f=poly(c,1); e=abs(f@np.linalg.lstsq(f,Y,rcond=None)[0]-Y); o=np.argsort(e[:,0])[-12:];print(np.round(np.c_[u[o],c[o],y[o],e[o,0]],3))
