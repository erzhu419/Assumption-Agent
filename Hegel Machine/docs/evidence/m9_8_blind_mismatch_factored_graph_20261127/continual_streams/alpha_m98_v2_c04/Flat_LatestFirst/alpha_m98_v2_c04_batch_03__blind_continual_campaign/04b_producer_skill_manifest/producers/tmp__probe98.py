import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_s00_solver as s
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr=d['train']; te=d['test']; dg=item['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; br=d['calibration_load_cells']; raw=br['bridge_raw']; true=br['true_channel_loads']; b=s._bridge(br)
xb=s._features(tr,te); x=xb[:len(tr['lab_id'])]; p,h=s._loads(tr,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; tau,theta=s._fit(x,p,h,dt,t,y,r)
print('tau',tau,'theta',theta); print('linear bridge rms, max by channel',np.sqrt(np.mean((np.c_[np.ones(len(raw)),raw]@b-true)**2,0)),np.max(np.abs(np.c_[np.ones(len(raw)),raw]@b-true),0))
for k in br['single_channel_sweeps']: print('sweep',k,np.shape(br['single_channel_sweeps'][k]),np.asarray(br['single_channel_sweeps'][k]).ravel()[:10])
for name in ['ref_coupons','lab_id','stratum','coupon_features_raw']:
 pass
joint={k:np.concatenate([te[k],dg[k]]) for k in ['lab_id','stratum','coupon_features_raw','ref_coupons']}; xd=s._features(tr,joint)[-len(dg['lab_id']):]
for name,bl,xx in [('train',tr,x),('diag',dg,xd)]:
 p,h=s._loads(bl,b); yy=bl['strain_curves_obs']-bl['strain_0'][:,None]; rr=bl['instant_rate_obs']; aa,bb=s._design(xx,p,h,dt,t,tau); ec=(aa@theta).reshape(yy.shape)-yy; er=bb@theta-rr; print(name,'curve and rate |errors| p50 p90 max',np.percentile(abs(ec),[50,90,100]),np.percentile(abs(er),[50,90,100])); print('curve residual column median',np.median(ec,0)); print('drift/coupon residual from curve-rate',np.round(np.c_[bl['lab_id'],ec[:,0],ec[:,-1],er],4)[:20]); print('past hold std',p.std(0),h.std())
