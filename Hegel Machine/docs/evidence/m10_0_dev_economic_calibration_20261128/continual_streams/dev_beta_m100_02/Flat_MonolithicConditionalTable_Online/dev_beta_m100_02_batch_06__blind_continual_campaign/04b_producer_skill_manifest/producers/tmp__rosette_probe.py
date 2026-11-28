import numpy as np,pickle
from scipy.optimize import minimize_scalar
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; ss=c['single_bank_sweeps']; sig=ss['ref_signal']; obs=ss['raw_readout']; bank=ss['bank_id']; clean=np.max(abs(y),axis=1)<5; iso=np.count_nonzero(x,axis=1)==1
print('pulse availability',item.get('campaign_diagnostics') is not None,'sweep reject',np.flatnonzero(~clean),'geometry',s['nx'],s['ny'],'modes',s['source_modes'].shape,'shifts',s['mu_shifts'])
funcs={'tanh':lambda z,a: np.tanh(a*z),'rational':lambda z,a:z/(1+a*z*z),'arctan':lambda z,a:np.arctan(a*z),'sqrt':lambda z,a:z/np.sqrt(1+a*z*z)}
for j in range(4):
 q=(bank==j)&(abs(obs)<5); u=(clean&iso)&(x[:,j]!=0); z=np.r_[sig[q],x[u,j]]; v=np.r_[obs[q],y[u,j]]; print('\nbank',j,'size',len(z))
 for name,f in funcs.items():
  for extras in ([],[1],[3],[1,3]):
   def fit(a):
    D=np.column_stack([np.ones(len(z)),f(z,a)]+[z**k for k in extras]); b=np.linalg.lstsq(D,v,rcond=None)[0]; return np.linalg.norm(D@b-v),b
   a=minimize_scalar(lambda a:fit(a)[0],bounds=(.08,5),method='bounded',options={'xatol':1e-13}).x; err,b=fit(a); print(name,extras,round(a,6),f'{err:.4g}',b.round(6))
 for k in range(4):
  print('isolated output',k,np.c_[x[u,j],y[u,k]].round(5).tolist())
