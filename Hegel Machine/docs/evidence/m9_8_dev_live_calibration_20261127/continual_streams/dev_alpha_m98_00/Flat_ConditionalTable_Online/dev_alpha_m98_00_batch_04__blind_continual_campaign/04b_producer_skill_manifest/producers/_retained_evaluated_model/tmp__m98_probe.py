import json,pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; C=D['calibration_load_cells']; Y=C['bridge_raw']; X=C['true_channel_loads']; np.set_printoptions(precision=4,suppress=True,linewidth=160)
idx=[()]+[i for j in range(1,4) for i in combinations_with_replacement(range(4),j)]; A=np.column_stack([np.prod(X[:,i],axis=1) for i in idx]);
mask=np.max(abs(Y),axis=1)<8; B=np.linalg.lstsq(A[mask],Y[mask],rcond=None)[0]
for i in range(6):
 E=Y-A@B; mask=np.max(abs(E),axis=1)<max(0.002,0.1/3**i); B=np.linalg.lstsq(A[mask],Y[mask],rcond=None)[0]; print('retained',mask.sum(),'rms',np.sqrt(np.mean((Y[mask]-A[mask]@B)**2)))
for i,b in zip(idx,B):
 if np.max(abs(b))>0.001:print(i,b)
E=Y-A@B; u,s,v=np.linalg.svd(E,full_matrices=False); print('singular',s,'direction',v[0]);print('unretained',np.c_[np.where(~mask)[0],(E@v[0])[~mask],np.linalg.norm(E-(E@v[0])[:,None]*v[0],axis=1)[~mask]])
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 z=np.array(e['facility_telemetry_trace']); print(e['completed_short_name'],'mean',z.mean(0),'sd',z.std(0)); print('initial/final',z[[0,-1]])
