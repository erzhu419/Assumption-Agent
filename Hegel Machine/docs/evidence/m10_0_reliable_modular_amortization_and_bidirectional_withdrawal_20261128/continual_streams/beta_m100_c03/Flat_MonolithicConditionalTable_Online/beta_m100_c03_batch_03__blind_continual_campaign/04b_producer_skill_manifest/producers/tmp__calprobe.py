import pickle,itertools,json,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X=c['ref_rosette']; Y=c['raw_rosette']
def basis(X,d):
 m=[()] + [z for j in range(1,d+1) for z in itertools.combinations_with_replacement(range(X.shape[1]),j)]
 return np.column_stack([np.prod(X[:,z],axis=1) for z in m]),m
def fit(A,B,seed=436):
 rng=np.random.default_rng(seed); n,p=A.shape; best=None
 for i in range(380):
  sub=rng.choice(n,min(n,max(p+2,int(.72*n))),replace=False); b=np.linalg.lstsq(A[sub],B[sub],rcond=None)[0]; e=np.linalg.norm(A@b-B,axis=1); ids=np.argsort(e)[:int(.85*n)]; score=np.linalg.norm(e[ids]);
  if best is None or score<best[0]: best=(score,b)
 b=best[1]
 for _ in range(8):
  e=np.linalg.norm(A@b-B,axis=1); keep=e<max(5*np.median(e),1e-10); b=np.linalg.lstsq(A[keep],B[keep],rcond=None)[0]
 return b
if __name__=='__main__':
 for d in (1,2,3):
  A,m=basis(X,d); b=fit(A,Y); e=np.linalg.norm(A@b-Y,axis=1); print('deg',d,'residual quantiles',np.quantile(e,[0,.5,.75,.9,.95,1]),'large rows',np.where(e>max(5*np.median(e),1.e-8))[0]); print([(k,np.round(row,7).tolist()) for k,row in zip(m,b) if max(abs(row))>0.0001]);
 a=c['single_bank_sweeps'];
 for k in range(4):
  j=a['bank_id']==k; x=a['ref_signal'][j]; y=a['raw_readout'][j]; A=np.vander(x,4,increasing=True); b=fit(A,y[:,None]);print('bank',k,b.ravel(),np.round(y-A@b.ravel(),8))
 for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
  t=np.asarray(a['facility_telemetry_trace']);print(a['completed_campaign_short_name'],t.mean(0).round(4),t.std(0).round(4))
 print('obs range',s['y_obs_raw'].min(),s['y_obs_raw'].max());print('reference range',X.min(),X.max())
