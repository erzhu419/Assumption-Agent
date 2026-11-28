import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as m
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; clean=(abs(y)<8).all(1); b=c['single_bank_sweeps']; ids,t,z=(b[k] for k in ('bank_id','ref_signal','raw_readout')); bank=abs(z)<8; ids,t,z=ids[bank],t[bank],z[bank]; opp=np.roll(np.arange(4),2)
def fit(keep,dp,dc,robust):
 X,Y=x[keep],y[keep]; V=X[:,opp]; p0=np.r_[np.zeros(4),np.ones(4),np.zeros(dp-1),np.zeros(dc)];
 def predict(p,Q):
  own=np.polynomial.polynomial.polyval(Q**2,np.r_[1,p[8:7+dp]])*Q
  cross=np.polynomial.polynomial.polyval(Q[:,opp],np.r_[0,p[7+dp:]])
  return p[:4]+p[4:8]*(own+cross)
 def residual(p):
  selfpred=p[:4][ids]+p[4:8][ids]*t*np.polynomial.polynomial.polyval(t*t,np.r_[1,p[8:7+dp]])
  return np.r_[(predict(p,X)-Y).ravel(),selfpred-z]
 p=least_squares(residual,p0,loss='soft_l1' if robust else 'linear',f_scale=4e-5,gtol=1e-12,max_nfev=250).x
 return lambda v:predict(p,np.atleast_2d(v)),p
base=m._calibrate(s)
for dp,dc,rob in [(5,8,False),(5,8,True),(6,8,True),(7,8,True),(6,10,True)]:
 pred,p=fit(clean,dp,dc,rob); fiterr=pred(x)-y; e=[]; small=[]
 for j in np.flatnonzero(clean):
  keep=clean.copy(); keep[j]=False; f,_=fit(keep,dp,dc,rob); r=f(x[j])[0]-y[j]; e.extend(r); small.extend(r[np.abs(x[j])<.65])
 obs=m._invert(pred,s['y_obs_raw']); print(dp,dc,rob,'rms/max',np.sqrt(np.mean(np.square(e))),max(abs(np.array(e))),'operating rms/max',np.sqrt(np.mean(np.square(small))),max(abs(np.array(small))),'cal change',max(abs(obs-base)),'coef',p.round(7));
 if dp==6 and dc==8: print('residual extremes',[(int(j),x[j].round(3).tolist(),fiterr[j].round(6).tolist()) for j in np.flatnonzero(clean)[np.argsort(np.max(abs(fiterr[clean]),axis=1))[-8:]]]);
