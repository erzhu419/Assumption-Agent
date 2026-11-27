import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
C=D['calibration_load_cells']; x=C['true_channel_loads']; y=C['bridge_raw']; good=np.max(abs(y-np.median(y,axis=0)),axis=1)<8
print('gross calibration rows',np.flatnonzero(~good)); x,y=x[good],y[good]
np.set_printoptions(precision=6,suppress=True)
rng=np.random.default_rng(72); folds=np.array_split(rng.permutation(len(x)),5)
lin=[(j,) for j in range(4)]; quads=list(combinations_with_replacement(range(4),2)); cubes=list(combinations_with_replacement(range(4),3))
models={'affine':lin,'self-cubic':lin+[(j,j,j) for j in range(4)], 'self-quad-cubic':lin+[(j,j) for j in range(4)]+[(j,j,j) for j in range(4)], 'cross-quad+self-cubic':lin+quads+[(j,j,j) for j in range(4)],'total cubic':lin+quads+cubes}
for name,terms in models.items():
 a=np.column_stack([np.ones(len(x))]+[np.prod(x[:,idx],axis=1) for idx in terms]); e=np.empty_like(y)
 for f in folds:
  t=np.setdiff1d(np.arange(len(x)),f); e[f]=a[f]@np.linalg.lstsq(a[t],y[t],rcond=None)[0]-y[f]
 print(name,'CV RMSE',np.sqrt(np.mean(e**2,axis=0)),'max',abs(e).max())
 if name in ('self-cubic','self-quad-cubic','cross-quad+self-cubic'):
  b=np.linalg.lstsq(a,y,rcond=None)[0]; print('coeffs',list(zip([()]+terms,b.round(6).tolist())))
print('past_dt', D['past_dt'])
