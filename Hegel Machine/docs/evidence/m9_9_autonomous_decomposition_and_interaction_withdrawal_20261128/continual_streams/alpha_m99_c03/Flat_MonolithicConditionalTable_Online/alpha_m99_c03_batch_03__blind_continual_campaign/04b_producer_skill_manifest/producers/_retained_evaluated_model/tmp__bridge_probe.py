import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
np.set_printoptions(precision=4,suppress=True,linewidth=140)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; r=c['bridge_raw']; y=c['true_channel_loads']; n=len(y); a=np.c_[np.ones(n),y]; fit=np.linalg.lstsq(a,r,rcond=None)[0]; e=r-a@fit; _,s,v=np.linalg.svd(e,full_matrices=False); q=e@v[0]
b=np.c_[np.ones(n),r]; inv=np.linalg.lstsq(b,y,rcond=None)[0]; err=b@inv-y
print('forward residual singular values',s,'dominant direction',v[0]); print('rows: index, measured load, dominant residual, orthogonal norm, affine axial residual'); ids=np.unique(np.r_[np.arange(0,n,3),np.argsort(abs(q))[-10:]]); print(np.c_[np.arange(n),y,q,np.linalg.norm(e-q[:,None]*v[0],axis=1),err.sum(1)][ids]); print('single sweep',np.c_[c['single_channel_sweeps']['lab_id'],c['single_channel_sweeps']['true_stress'],c['single_channel_sweeps']['cmd_stress']]);
def terms(x,deg): return np.column_stack([np.ones(len(x))]+[x[:,idx].prod(1) for j in range(1,deg+1) for idx in cwr(range(x.shape[1]),j)])
for blocked in (False,True):
 results={k:np.empty(n) for k in ('affine','quadratic','projected affine','trimmed forward','trimmed inverse','robust inverse')}
 for f in range(5):
  te=(np.arange(n)//(n//5) if blocked else np.arange(n)%5)==f; tr=~te
  def pred(name,x,keep): results[name][te]=x[te]@np.linalg.lstsq(x[keep],y[keep].sum(1),rcond=None)[0]
  pred('affine',b,tr); pred('quadratic',terms(r,2),tr)
  m=np.linalg.lstsq(a[tr],r[tr],rcond=None)[0]; _,_,v=np.linalg.svd(r[tr]-a[tr]@m,full_matrices=False); pred('projected affine',terms(r@v[1:].T,1),tr)
  keep=tr.copy()
  for _ in range(10):
   m=np.linalg.lstsq(a[keep],r[keep],rcond=None)[0]; norm=np.linalg.norm(r-a@m,axis=1); order=np.flatnonzero(tr); keep=np.zeros(n,bool); keep[order[np.argsort(norm[tr])[:int(.75*tr.sum())]]]=True
  m=np.linalg.lstsq(a[keep],r[keep],rcond=None)[0]; results['trimmed forward'][te]=np.linalg.solve(m[1:].T,(r[te]-m[0]).T).sum(0); pred('trimmed inverse',b,keep)
  from scipy.optimize import least_squares
  z=y[tr].sum(1); initial=np.linalg.lstsq(b[tr],z,rcond=None)[0]; results['robust inverse'][te]=b[te]@least_squares(lambda p:b[tr]@p-z,initial,loss='soft_l1',f_scale=.06).x
 print('blocked',blocked)
 for k,p in results.items():
  z=p-y.sum(1); print(k,'RMSE',np.mean(z*z)**.5,'max',max(abs(z)),'20-row RMSE',np.sqrt((z.reshape(4,-1)**2).mean(1)))
