import pickle,numpy as np
from itertools import combinations_with_replacement
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; y=C['true_channel_loads']; r=C['bridge_raw']; sw=C['single_channel_sweeps']; ps=[()]+[p for k in range(1,4) for p in combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(y[:,p],axis=1) for p in ps]); B=np.linalg.lstsq(A,r,rcond=None)[0]; v=np.linalg.svd(r-A@B,full_matrices=False)[2][0]; P=np.eye(4)-np.outer(v,v); print('direction',v,'sum',v.sum()); print('projected significant terms:');
for p,b in zip(ps,B@P):
 if np.max(np.abs(b))>.001:print(p,b.round(5))
print('scalar nuisance residual vs true load channel 3'); e=(r-A@B)@v
for j in np.argsort(y[:,3]):print(j,round(y[j,3],4),round(e[j],5))
print('sweeps');
for j in range(4):
 ix=sw['lab_id']==j; print(j,np.c_[sw['true_stress'][ix],sw['cmd_stress'][ix]].round(5).T)
print('isolated channel 3 observed');print(r[12:16]);
