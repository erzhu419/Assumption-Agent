import sys,glob,pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c03_s00_solver as m
u=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=u['dataset']; tr=d['train']; te=d['test']; dg=u['campaign_diagnostics']; dt=d['past_dt']; t=d['eval_times']; a,b=m._prepare(d)
def target(g): return np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
y=target(tr); yd=target(dg); tau,w=m._fit(a,dt,t,y); print('tau/w',tau,w); off,tot=m._calibration(d['calibration_load_cells']); z=m._features(tr,{k:np.concatenate([te[k],dg[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')})[-len(yd):]; c=((dg['past_channel_raw']-off)@tot,(dg['hold_channel_raw']-off)@tot,np.c_[np.ones(len(z)),z]); p=m._design(c,dt,t,tau)@w

def stats(r): return np.round(np.sqrt(np.mean(r*r,axis=0)),5)
print('train RMSE',stats(m._design(a,dt,t,tau)@w-y)); print('diag RMSE',stats(p-yd)); print('diag first rows obs/err',np.round(yd[:3],4),np.round((yd-p)[:3],4))
def basis(taus):return np.vstack([np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1.]])
B=basis(tau); A=np.linalg.lstsq(B,yd.T,rcond=None)[0].T; print('individual shared tau RMSE',stats(A@B.T-yd));print('median abs',np.median(abs(A@B.T-yd),0));
def fun(q):
 h=basis(np.exp(q)); return (h@np.linalg.lstsq(h,yd.T,rcond=None)[0]-yd.T).ravel()
f=least_squares(fun,np.log(tau),loss='soft_l1',f_scale=.005,bounds=(np.log(.051),np.log(24.99))); print('free chamber tau',np.exp(f.x),'rmse',stats(f.fun.reshape(len(t)+1,-1).T));
base=m._design(c,dt,t,tau); A0=np.c_[base[:,-1,:2]*tau*w[:2],c[2]@w[2:]]; print('inferred coefficient rows [baseline,observed,past,hold]'); print(np.round(np.c_[A0,A,c[0],c[1]],3)); print('reference definition diff sensitivity'); print('diag telemetry?',dg.keys());print('test telemetry mean/sd',te['aux_chamber_telemetry'].mean(0),te['aux_chamber_telemetry'].std(0));
