import pickle,itertools,numpy as np
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
def F(x,d=1):
 return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in itertools.combinations_with_replacement(range(4),k)])
for kp,kv in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[kp]; u=p['u']; v=p[kv]; f=F(u); keep=np.ones(len(u),bool)
 if kp=='panel_a':
  for _ in range(5):
   w=lstsq(f[keep],v[keep])[0]; er=np.linalg.norm(v-f@w,axis=1); keep=er<=np.median(er)*3
 print(kp,kv,'kept',keep.sum(),'linear coef',lstsq(f[keep],v[keep])[0]);
 for d in (1,2,3):
  f=F(u,d)[keep]; vv=v[keep]; e=vv-f@lstsq(f,vv)[0]; h=np.sum(np.linalg.svd(f,full_matrices=False)[0]**2,axis=1); print(d,'rms',np.sqrt(np.mean(e**2,axis=0)),'loo mae',np.mean(abs(e/np.maximum(1-h[:,None],1e-12)),axis=0) if len(f)>f.shape[1] else None)
 if kp=='panel_a':
  for j in range(4):
   k=(j+1)%4; a,b=u[:,j],u[:,k]; f=np.c_[np.ones(len(u)),a,b,a*b,a**3]; w=lstsq(f[keep],v[keep,j])[0]; print('cyclic',j,w,'rms',np.std((v[:,j]-f@w)[keep]))
x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; print('sv',np.linalg.svd(tr,compute_uv=False)); print('traj avg',tr.mean(0),'y std',y.std());
for d in (1,2,3):
 f=F(x,d); Y=np.c_[y,tr]; e=Y[56:]-f[56:]@lstsq(f[:56],Y[:56])[0]; print('raw holdout',d,np.mean(abs(e),axis=0))
