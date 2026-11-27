import itertools as it,pickle,numpy as np,json
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; np.set_printoptions(precision=6,suppress=True)
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 t=np.array(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'],t.shape,t.mean(0),t.std(0))
print('axis rows',np.c_[x,y][:16]); print('raw max by row',np.round(abs(y).max(1),3));
for d in range(1,6):
 terms=[()] + [a for k in range(1,d+1) for a in it.combinations_with_replacement(range(4),k)]; A=np.column_stack([x[:,a].prod(1) for a in terms]); use=abs(y).max(1)<5
 if A.shape[1]>=use.sum():break
 p=np.linalg.lstsq(A[use],y[use],rcond=None)[0]; err=y-A@p; print('poly',d,'rows',use.sum(),'rms',np.sqrt(np.mean(err[use]**2)),'max',abs(err[use]).max(),'condition',np.linalg.cond(A[use]))
 for a,v in zip(terms,p):
  if abs(v).max()>.015: print(a,v)
