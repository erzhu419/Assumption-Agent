import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def poly(x,d=2):
 return np.column_stack([np.prod(x[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
if __name__=='__main__':
 print('time',s['eval_times'],'telemetry',s['telemetry_vec'])
 for k in ['panel_a','panel_b','panel_c']:
  p=s[k]; u=p['u']; y=np.column_stack([a for q,a in p.items() if q!='u']); print(k)
  for d in (1,2):
   X=poly(u,d); B=fit(X,y); e=y-X@B; h=np.sum(np.linalg.qr(X,mode='reduced')[0]**2,axis=1); print(d,'loo MAE',np.mean(abs(e/(1-h[:,None])),axis=0),'row RMS sorted',np.sort(np.mean(e**2,axis=1)**.5)); print('coef',B if d==1 else np.linalg.norm(B,axis=1))
 x=s['train']['x']; z=s['train']['traj']; y=s['train']['y']; print('traj sv',np.linalg.svd(z,compute_uv=False)); print('traj means',z.mean(0)); print('traj diff sv',np.linalg.svd(np.diff(z,axis=1),compute_uv=False)); print('y-from-traj',np.mean(abs(y-poly(z,1)@fit(poly(z,1),y)))); print('traj residual per time from rank 2/3',*[np.std(z-np.linalg.svd(z,full_matrices=False)[0][:,:j]@np.diag(np.linalg.svd(z,compute_uv=False)[:j])@np.linalg.svd(z,full_matrices=False)[2][:j],axis=0) for j in [2,3]])
 for d in (1,2,3):
  X=poly(x,d); Y=np.c_[y,z]; B=fit(X[:56],Y[:56]); print('raw poly',d,'heldout MAE',np.mean(abs(X[56:]@B-Y[56:]),axis=0))
