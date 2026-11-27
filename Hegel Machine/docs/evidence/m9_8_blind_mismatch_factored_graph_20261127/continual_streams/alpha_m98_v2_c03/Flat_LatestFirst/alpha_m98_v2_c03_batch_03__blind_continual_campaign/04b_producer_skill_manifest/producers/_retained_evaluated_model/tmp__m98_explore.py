import pickle,numpy as np
import m98_base as b
np.set_printoptions(precision=5,suppress=True)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; dg=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; train,test=b._prepare(d)
def y(g):return np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]
o,c=b._calibration(d['calibration_load_cells']); cal=d['calibration_load_cells']; L=cal['true_channel_loads']; B=cal['bridge_raw']; print('cal residual quantile',np.quantile(abs((B-o)@c-L.sum(1)),[.5,.9,1])); print('axes',dt,t)
x=b._features(tr,te); panels={k:np.r_[tr[k],te[k]] for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
def diag_x(g):
 labs=panels['lab_id']; f=g['coupon_features_raw'].copy();
 for lab in np.unique(g['lab_id']):
  refs=panels['ref_coupons'][labs==lab]; m=g['lab_id']==lab;f[m]=(f[m]-np.median(refs.mean(0)))/np.median(refs.std(0))
 for j in range(f.shape[1]):
  xx=x[:,j]; nan=np.isnan(panels['coupon_features_raw'][:,j]); f[np.isnan(f[:,j]),j]=xx[nan][0] if nan.any() else xx.mean()
 return f
D=diag_x(dg); diag=((dg['past_channel_raw']-o)@c,(dg['hold_channel_raw']-o)@c,np.c_[np.ones(len(D)),D]); taus,th=b._fit(train,dt,t,y(tr)); print('train',taus,th); print('train resid',np.quantile(abs(b._design(train,dt,t,taus)@th-y(tr)),[.5,.9,1],axis=0)); print('diag independent',b._fit(diag,dt,t,y(dg)))
F=np.r_[-np.expm1(-t[:,None]/taus),[1/taus]]; F=np.c_[F,np.r_[t,1]]; co=np.linalg.lstsq(F,y(dg).T,rcond=None)[0].T; a=b._design(diag,dt,t,taus); bas=np.c_[a[:,-1,:2]*taus*th[:2],diag[2]@th[2:]]; print('diag curve basis residual',np.quantile(abs(co@F.T-y(dg)),[.5,.9,1],axis=0)); print('amplitudes baseline and fit');print(np.c_[bas,co]);print('diag nominal residual',np.quantile(abs(a@th-y(dg)),[.5,.9,1],axis=0));print('diag stresses sample',np.c_[diag[0],diag[1]][:6]);print('sweeps',type(cal['single_channel_sweeps']),str(cal['single_channel_sweeps'])[:350]);print('missing diagnostic',np.isnan(dg['coupon_features_raw']).sum(0));print('pickle items',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
