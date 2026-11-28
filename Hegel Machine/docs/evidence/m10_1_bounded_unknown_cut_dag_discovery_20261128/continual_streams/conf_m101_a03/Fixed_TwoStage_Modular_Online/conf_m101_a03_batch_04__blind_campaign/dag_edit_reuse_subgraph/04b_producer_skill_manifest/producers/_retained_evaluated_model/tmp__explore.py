import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations as co, combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def cross(u):return np.column_stack([u[:,i]*u[:,j] for i,j in co(range(4),2)])
def feat(u):return np.c_[np.ones(len(u)),u,cross(u),u**3]
p=s['panel_a']; u,v=p['u'],p['v']; A=feat(u); w=np.ones(len(u))
for _ in range(40):
 b=np.linalg.lstsq(A*w[:,None],v*w[:,None],rcond=None)[0];e=np.linalg.norm(A@b-v,axis=1);w=(1+(e/max(np.median(e),1e-7))**2)**-.5
print('A',np.round(b.T,5), 'res sorted',np.round(np.sort(e)[-8:],5)); b[abs(b)<.002]=0
x=np.r_[s['train']['x'],s['test']['x']]; z=np.array([least_squares(lambda u:(feat(u[None])@b)[0]-row,x0).x for row,x0 in zip(x,np.linalg.lstsq(b[1:5].T,(x-b[0]).T,rcond=None)[0].T)]); print('Z ranges',z.min(0),z.max(0),'train',z[:72].min(0),z[:72].max(0),'max inverse residual',np.max(abs(feat(z)@b-x)))
for pk,vk,fun in [('panel_b','v',np.tanh),('panel_c','v1',np.tanh),('panel_c','v2',np.sin)]:
 p=s[pk]; u,v=p['u'],p[vk]; print(pk,vk,'v ranges',v.min(0),v.max(0))
 for mode in ('axis','output','outputcross','square','squarefixed'):
  def calc(a):
   if mode=='axis': X=np.c_[np.ones(len(u)),u,fun(a*u),cross(u)]; V=v
   if mode in ('output','outputcross'):
    if a<=abs(v).max(): return 1e3,None
    X=np.c_[np.ones(len(u)),u] if mode=='output' else np.c_[np.ones(len(u)),u,cross(u)]; V=np.arctanh(v/a)
   if mode=='square': X=np.c_[np.ones(len(u)),u,u**2]; V=v
   if mode=='squarefixed':X=np.c_[np.ones(len(u)),u+a*u**2];V=v
   B=np.linalg.lstsq(X,V,rcond=None)[0]; R=(np.tanh(X@B)*a if mode.startswith('output') else X@B)-v
   return np.sqrt(np.mean(R**2)),B
  r=minimize_scalar(lambda a: calc(a)[0],bounds=((abs(v).max()+1e-5,5) if mode.startswith('output') else (.1,4)),method='bounded'); print(mode,round(r.x,5),round(r.fun,6))
Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; print('times',t,'sv',np.linalg.svd(Y[:,1:],compute_uv=False)); print('corr linear',np.corrcoef(np.c_[z[:72],Y].T)[:4,4:][:,[0,1,-1]])
for name,u in [('raw',x[:72]),('inv',z[:72])]:
 for degree in range(1,5):
  P=np.column_stack([np.ones(len(u))]+[np.prod(u[:,i],axis=1) for j in range(1,degree+1) for i in cr(range(4),j)]); tr=np.arange(52); te=np.arange(52,72)
  for l in ([0] if degree<3 else [.01,.1,1]):
   B=np.linalg.lstsq(np.r_[P[tr],np.sqrt(l)*np.eye(P.shape[1])],np.r_[Y[tr],np.zeros((P.shape[1],9))],rcond=None)[0]; E=abs(Y[te]-P[te]@B); print(name,degree,l,np.round([E[:,0].mean(),E[:,1:].mean(),E[:, -1].mean()],5))
np.savez('/tmp/panelcoords.npz',z=z,x=x)
