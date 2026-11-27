import pickle,numpy as np
import rheo_base as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; g=p['campaign_diagnostics']; tr=d['train']; te=d['test']; tt=d['eval_times']; dt=d['past_dt']; q=m._calibration(d['calibration_load_cells']); n=len(tr['lab_id']); x=m._features(tr,te); joint={k:np.concatenate([te[k],g[k]]) for k in ['lab_id','stratum','ref_coupons','coupon_features_raw']}; gx=m._features(tr,joint)[-len(g['lab_id']):]
def coupon(s,x): return m._stress(s['past_channel_raw'],q),m._stress(s['hold_channel_raw'],q),np.c_[np.ones(len(x)),x]
def y(s): return np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
tc=coupon(tr,x[:n]); gc=coupon(g,gx); taus,b=m._fit(tc,dt,tt,y(tr)); h,H,X=gc; drift=X@b[2:]; design=m._design(gc,dt,tt,taus); pred=design@b; obs=y(g); gap=design[:,-1,:2]*taus
gap=design[:,-1,:2]*taus
from scipy.optimize import least_squares
F=np.c_[np.r_[-np.expm1(-tt[:,None]/taus),[1/taus]],np.r_[-np.expm1(-tt),1.]]
coeff=np.array([least_squares(lambda c:F@c-row,np.linalg.lstsq(F,row,rcond=None)[0],loss='soft_l1',f_scale=.004).x for row in obs]); zg=H[:,None]-gap
print('coeff/gaps: lab H z1 z2 A1/theta A2/theta D drift'); print(np.c_[g['lab_id'],H,zg,coeff[:,:2]/b[:2],coeff[:,2],drift].round(3))
print('projection q90',np.quantile(abs(coeff@F.T-obs),.9));
rev=m._design((h[:,::-1],H,X),dt,tt,taus)[:,-1,:2]*taus
rev2=m._design((h[:,::-1],H,X),dt[::-1],tt,taus)[:,-1,:2]*taus
for j in range(3):
 yy=coeff[:,j]/(b[j] if j<2 else 1)
 z=zg[:,min(j,1)]; ga=H-z; R=rev[:,min(j,1)]; R2=rev2[:,min(j,1)]; one=np.ones(len(H));
 print('branch',j)
 for label,V in [('H',H),('gap',ga),('reverse',R),('reverse2',R2),('drift',drift),('H+z',np.c_[H,z]),('1+H+z',np.c_[one,H,z]),('linear past',np.c_[one,H,h]),('linear+drift',np.c_[one,H,h,drift]),('gap+H^2',np.c_[ga,H**2]),('1+drift',np.c_[one,drift]),('1+X',X)]:
  V=np.asarray(V).reshape(len(yy),-1); w=np.linalg.lstsq(V,yy,rcond=None)[0]; e=yy-V@w; cv=yy.copy();
  for l in np.unique(g['lab_id']):
   mask=g['lab_id']==l; cv[mask]-=V[mask]@np.linalg.lstsq(V[~mask],yy[~mask],rcond=None)[0]
  print(label,'fit/cv rms',np.sqrt(np.mean(e**2)).round(4),np.sqrt(np.mean(cv**2)).round(4),'coef',w.round(3))
