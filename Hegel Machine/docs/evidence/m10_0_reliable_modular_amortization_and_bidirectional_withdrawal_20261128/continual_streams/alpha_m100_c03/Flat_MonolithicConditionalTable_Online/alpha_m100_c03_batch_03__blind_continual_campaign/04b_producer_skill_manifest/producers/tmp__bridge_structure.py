import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; y=C['true_channel_loads']; r=C['bridge_raw']; ps=[()]+[p for k in range(1,4) for p in combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(y[:,p],axis=1) for p in ps]); S=np.count_nonzero(y,axis=1)==1; B=np.column_stack([np.ones(len(y)),y,y**3]); b=np.linalg.lstsq(B[S],r[S],rcond=None)[0]; e=r-B@b; print('isolated coeffs',b.round(5)); print('isolated max error',abs(e[S]).max(axis=0),'singular values',np.linalg.svd(e,compute_uv=False)); print('largest residual rows: true, raw minus isolated fit');
for i in np.argsort(np.linalg.norm(e,axis=1))[-15:]:print(i,y[i].round(4),e[i].round(4))
for k in range(4):
 n=np.linalg.lstsq(A,r[:,k],rcond=None)[0]; print('channel',k,'residual singular/full',np.std(A@n-r[:,k]),'sum clean cubic terms') if k==0 else None
n=np.linalg.lstsq(A,r.sum(1),rcond=None)[0]; print([(p,round(v,6)) for p,v in zip(ps,n) if abs(v)>.001]); F=r-A@np.linalg.lstsq(A,r,rcond=None)[0]; print('full cubic residual SVD',np.linalg.svd(F,full_matrices=False)[1:]);
for lim in [.001,.01,.1,.5,1,2]:
 m=np.max(abs(e),axis=1)<lim; print('max residual gate',lim,'n',sum(m),'mixed',sum(m&~S),'rank cubic',np.linalg.matrix_rank(A[m]) if m.any() else 0)
print('all calibration row blocks (nonzero true channels, sum/raw residual norm):');
for i in range(0,len(y),8):print(i,np.count_nonzero(y[i:i+8],axis=1),np.round(np.linalg.norm(e[i:i+8],axis=1),3));
