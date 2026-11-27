import pickle,numpy as np
import rheo_base as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; g=p['campaign_diagnostics']; tr=d['train']; te=d['test']; tt=d['eval_times']; dt=d['past_dt']; q=m._calibration(d['calibration_load_cells']); n=len(tr['lab_id']); x=m._features(tr,te); joint={k:np.concatenate([te[k],g[k]]) for k in ['lab_id','stratum','ref_coupons','coupon_features_raw']}; gx=m._features(tr,joint)[-len(g['lab_id']):]
def coupon(s,x): return m._stress(s['past_channel_raw'],q),m._stress(s['hold_channel_raw'],q),np.c_[np.ones(len(x)),x]
def y(s): return np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
tc=coupon(tr,x[:n]); gc=coupon(g,gx); taus,b=m._fit(tc,dt,tt,y(tr)); h,H,X=gc; drift=X@b[2:]; design=m._design(gc,dt,tt,taus); pred=design@b; obs=y(g); gap=design[:,-1,:2]*taus; print('taus,theta',taus,b)
F=np.r_[-np.expm1(-tt[:,None]/taus),[1/taus]]; L=np.r_[tt,1.]; S=np.r_[-np.expm1(-tt),1.]; print('per horizon observed vs baseline median',np.median(obs,axis=0).round(3),np.median(pred,axis=0).round(3)); print('med abs residual by horizon',np.median(abs(obs-pred),axis=0).round(3));
for nm,D in [('linear',L),('sat',S)]:
 for mode in ['free-branches','free-all','free-drift']:
  A={'free-branches':F,'free-all':np.c_[F,D],'free-drift':D[:,None]}[mode]; offset={'free-branches':drift[:,None]*D,'free-all':0,'free-drift':design[:,:,:2]@b[:2]}[mode]; coef=(obs-offset)@np.linalg.pinv(A).T; e=obs-offset-coef@A.T; print(nm,mode,'mad',np.median(abs(e)),'q90',np.quantile(abs(e),.9));
  if mode=='free-branches': print(np.c_[g['lab_id'],H,h,drift,gap,coef/b[:2],obs[:,-1]-pred[:,-1],np.median(abs(e),axis=1)].round(3))
for i in range(4):
 a=d['calibration_load_cells']['single_channel_sweeps']; mask=a['lab_id']==i; print('sweep',i,np.c_[a['true_stress'][mask],a['cmd_stress'][mask]].round(4));
print('raw range',[(m,np.quantile(abs(v),[.5,.9,.99,1])) for m,v in [('ch diag',g['hold_channel_raw']),('past diag',g['past_channel_raw']),('test hold',te['hold_channel_raw']),('test past',te['past_channel_raw'])]])
