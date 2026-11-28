import sys,pickle
sys.path.insert(0,'/holdout_task')
from node_00 import _responses,_invert
from scipy.optimize import least_squares
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; z=_invert(s['panel_a'],x); b,c,d=_responses(s,z); n=len(z); t=s['train']['traj']; basis=np.linalg.svd(t,full_matrices=False)[2][:2]; y=np.column_stack((s['train']['y'], t@basis.T)); one=np.ones(n)
print('inv',np.round(z.min(0),3),np.round(z.max(0),3)); print('panel bounds',[(a.min(),a.max()) for a in (b,c,d)])
def outer(a,b):return (a[:,:,None]*b[:,None,:]).reshape(n,-1)
splits=[(np.arange(52),np.arange(52,72)),(np.arange(18,72),np.arange(18)),(np.r_[0:18,36:72],np.arange(18,36))]
results=[]
for func in (np.tanh,np.sin):
 for k in (.75,1,1.35,1.7,2,2.5,3):
  for name,extra in [('none',[]),('diag',[b*c]),('diagC',[b*c,c*d]),('full',[outer(b,c)]),('fullD',[outer(b,d)]),('cd',[outer(c,d)]),('mix',[b*c,b*d,c*d]),('zt',[outer(z,b)]),('zd',[z*b,z*c,z*d]),('fuse',[outer(b,c),d]),('c',[c,d]),('tan',[np.tanh(2*c),np.tanh(2*d)]),('tanz',[np.tanh(2*z)]),('fullt',[outer(func(k*b),c)]),('fulltd',[outer(func(k*b),d)])]:
   for p in (.0003,.01,.06):
    h=np.column_stack((one,z,func(k*b),*extra)); rr=[]
    for tr,te in splits:
     g=h[tr]; pen=np.eye(g.shape[1])*p;pen[0,0]=0; pred=h[te]@np.linalg.solve(g.T@g+pen,g.T@y[tr]); rr.append([abs(pred[:,0]-y[te,0]).mean(),abs(pred[:,1:]@basis-t[te]).mean()]);
    results.append((max(np.array(rr).ravel()),func.__name__,k,name,p,np.round(rr,4).tolist()))
for r in sorted(results)[:14]:print(r)
print('public only')
for r in sorted(results,key=lambda r:max(r[-1][0]))[:8]:print(r)
