import pickle,json,numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=135)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; D=np.c_[np.ones(len(x)),x]; B=np.linalg.lstsq(D,y,rcond=None)[0]; B=least_squares(lambda b:(D@b.reshape(5,4)-y).ravel(), B.ravel(),loss='cauchy',f_scale=.02).x.reshape(5,4); R=y-D@B
print('affine robust coefficients\n',B); print('row residual norms',np.linalg.norm(R,axis=1).round(4)); print('first 16 residuals\n',R[:16]); print('largest residual rows');
for i in np.argsort(np.linalg.norm(R,axis=1))[-8:]: print(i,x[i],R[i])
b=c['single_bank_sweeps']; print('single bank shapes',[(k,v.shape) for k,v in b.items()]); print('single bank first',[(k,v[:6]) for k,v in b.items()]); print('obs range',s['y_obs_raw'].min(),s['y_obs_raw'].max()); print('obs bank pattern',s['row_bank_id'][:32]); print('sensor indices',s['sensor_ij']); print('aux moments',s['aux_manifold_telemetry'].mean(0),s['aux_manifold_telemetry'].std(0));
for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.array(r['facility_telemetry_trace']); print(r['completed_campaign_short_name'],'mean',a.mean(0),'std',a.std(0))
