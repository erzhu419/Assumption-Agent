import pickle,json,importlib.util,numpy as np
from itertools import combinations_with_replacement
p='/tmp/hegel_continual_archive/'
spec=importlib.util.spec_from_file_location('old',p+'beta_m99_c02_s00_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];s=item['study']; c=s['sensor_calibration_sweeps'];x,y=c['ref_rosette'],c['raw_rosette']; ds=item['campaign_diagnostics']; W=np.stack([d['w_pulse'] for d in ds],axis=1); Y=np.stack([d['y_pulse_raw'] for d in ds],axis=1)
np.set_printoptions(precision=6,suppress=True,linewidth=140)
for d in [1,2,3]:
 terms=[t for j in range(1,d+1) for t in combinations_with_replacement(range(4),j)]
 X=np.column_stack([np.ones(len(x))]+[x[:,t].prod(1) for t in terms]); b=np.linalg.lstsq(X,y,rcond=None)[0]; R=y-X@b
 print('degree',d,'rank',np.linalg.matrix_rank(X),'RMS',np.sqrt(np.mean(R**2,axis=0)),'max',np.max(abs(R)))
 if d==2:
  for t,v in zip([()]+terms,b): print(t,v)
 print('leave-group metrics', [round(float(np.sqrt(np.mean((y[i:i+8]-X[i:i+8]@np.linalg.lstsq(np.delete(X,np.s_[i:i+8],0),np.delete(y,np.s_[i:i+8],0),rcond=None)[0])**2))),7) for i in range(0,len(x),8)])
print('single-bank fits')
a=c['single_bank_sweeps'];
for j in range(4):
 m=a['bank_id']==j; z=a['ref_signal'][m]; v=a['raw_readout'][m];print(j,z,v,np.polynomial.polynomial.polyfit(z,v,1))
print('rosette sample groups',x[::4]);print('bank arrangement',s['row_bank_id'][:16]);print('sensors',s['sensor_ij']);
for e in json.load(open(p+'archive_apparatus_log.json')):
 a=np.asarray(e['facility_telemetry_trace']);print(e['completed_campaign_short_name'],np.mean(a,axis=0),np.std(a,axis=0),a[-1]);
a=s['aux_manifold_telemetry'];print('public aux',np.mean(a,axis=0),np.std(a,axis=0))
S,T=old._responses(s); b=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]; yp=np.stack([(S@w).reshape(-1,4)@b[1:]+b[0] for w in W.T],axis=-1).reshape(Y.shape);print('baseline pulse RMS by shift/pulse',np.sqrt(np.mean((yp-Y).reshape(3,-1,3)**2,axis=1)));print('fit pulse residual nuisance',np.sqrt(np.mean(old._profile(yp,Y[:,0],s)[0]**2)))
