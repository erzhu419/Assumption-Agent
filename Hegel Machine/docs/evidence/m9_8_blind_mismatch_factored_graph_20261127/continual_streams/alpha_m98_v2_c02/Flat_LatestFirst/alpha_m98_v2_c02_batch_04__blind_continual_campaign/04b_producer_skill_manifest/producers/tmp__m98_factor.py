import pickle, numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']; S=L.sum(1)
np.set_printoptions(precision=6,suppress=True,linewidth=145)
idx=[j for k in range(1,4) for j in combinations_with_replacement(range(4),k)]
X=np.column_stack([np.ones(len(L))]+[np.prod(L[:,j],axis=1) for j in idx]); A=np.linalg.lstsq(X,R,rcond=None)[0]; E=R-X@A
print('cubic residual singular values',np.linalg.svd(E,compute_uv=False)); print('cubic residual right basis',np.linalg.svd(E,full_matrices=False)[2]); print('forward coefficients: monomial, row, row sum');
for j,a in zip([()]+idx,A): print(j,a,a.sum())
for v in [S,L[:,0]-L[:,1],L[:,2]-L[:,3]]: print('projection onto loading coordinate quantiles',np.quantile(v,[0,.1,.5,.9,1]))
print('raw sums vs true total residual from cubic of total',np.linalg.lstsq(np.column_stack([S**k for k in range(4)]),R.sum(1),rcond=None)[0],np.std(R.sum(1)-np.polynomial.polynomial.polyval(S,np.linalg.lstsq(np.column_stack([S**k for k in range(4)]),R.sum(1),rcond=None)[0])))
