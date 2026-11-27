import importlib.util, pickle, numpy as np
from scipy.sparse import csc_matrix
spec=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/dev_beta_m99_00_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=it['study']; p=it['campaign_diagnostics']
W=np.column_stack([v['w_pulse'] for v in p]); Yraw=np.column_stack([v['y_pulse_raw'] for v in p]); cal=s['sensor_calibration_sweeps']; X=cal['ref_rosette']; Y=cal['raw_rosette']
B=np.linalg.lstsq(np.column_stack((np.ones(len(X)),X)),Y,rcond=None)[0]
def axis(k,v,h):
 n=len(k); a=np.zeros((n,n)); i=np.arange(n); prev=(i-1)%n; nxt=(i+1)%n
 lm=(k+k[prev])/(2*h*h)+np.maximum(v,0)/h
 lp=(k+k[nxt])/(2*h*h)+np.maximum(-v,0)/h
 a[i,prev]=-lm; a[i,nxt]=-lp; a[i,i]=lm+lp
 return csc_matrix(a)
for key in 'xy':
 k=s['kappa_'+key]; v=s['vel_'+key]; h=s['L'+key]/(len(k)-1); a=axis(k,v,h).toarray(); print(key,'exact endpoint deltas (inward, wrap)',[(round(-a[i,j]*h*h/k[i]-1,8),round(-a[i,o]*h*h/k[i],8)) for i,j,o in [(0,1,-1),(-1,-2,0)]],'constant error',np.max(np.abs(a.sum(axis=1))))
m._axis=axis
S,U=m._forward(s); Ypred=(S@W).T.reshape(3,-1,4)@B[1:]+B[0]; E=Ypred-Yraw.T.reshape(3,-1,4)
print('periodic pulse raw RMSE per pulse/shift',np.sqrt(np.mean(E.reshape(3,len(s['mu_shifts']),-1)**2,axis=2))); print('worst pulse/bank RMSE',np.sqrt(np.mean(E**2,axis=1)).max()); print('max abs pulse residual',np.abs(E).max())
