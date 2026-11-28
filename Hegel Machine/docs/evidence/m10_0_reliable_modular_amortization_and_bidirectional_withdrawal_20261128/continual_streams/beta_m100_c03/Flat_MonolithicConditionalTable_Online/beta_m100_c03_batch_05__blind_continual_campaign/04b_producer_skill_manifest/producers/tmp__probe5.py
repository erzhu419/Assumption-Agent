import pickle, json
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']
np.set_printoptions(precision=7,suppress=True,linewidth=140)
keep=np.max(abs(y),axis=1)<5
for j in range(4):
 a=np.r_[x[keep & (np.count_nonzero(x,axis=1)==1),j],q['ref_signal'][q['bank_id']==j]]; b=np.r_[y[keep & (np.count_nonzero(x,axis=1)==1),j],q['raw_readout'][q['bank_id']==j]]; a=a[abs(b)<5]; b=b[abs(b)<5]; use=(a!=0) | (abs(b-np.median(b[a==0]))<.02);a,b=a[use],b[use]
 for name,f in [('tanh',np.tanh),('softsign',lambda t:t/np.sqrt(1+t*t)),('atan',np.arctan)]:
  fun=lambda p:p[0]+p[1]*f(p[2]*a)-b; fit=least_squares(fun,[0,1.5,1.3]); print(j,name,fit.x,np.sqrt(np.mean(fun(fit.x)**2)))
print('archive channel 0');
for t in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(t['facility_telemetry_trace']);print(t['completed_campaign_short_name'],a[::4,0], 'last',a[-1,0])
print('sensors',s['sensor_ij'].tolist(),'shifts',s['mu_shifts'])
